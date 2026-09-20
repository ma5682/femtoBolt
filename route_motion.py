"""Distance-based research playback. No time or interpolation across missing links."""
import numpy as np


def statistics(plan,speed_mm_s=20.):
    if not np.isfinite(speed_mm_s) or speed_mm_s<=0:raise ValueError('Invalid simulation speed')
    scan=connector=0.;lengths=[]
    for chain in plan['segments']:
        edges=np.linalg.norm(np.diff(np.asarray([p['xyz_m'] for p in chain]),axis=0),axis=1)
        lengths.append(float(edges.sum()))
        for length,p in zip(edges,chain[1:]):
            if p.get('motion_kind','scan')=='connector':connector+=float(length)
            else:scan+=float(length)
    return dict(scan_length_m=scan,connector_length_m=connector,total_length_m=scan+connector,
        chain_count=len(lengths),chain_lengths_m=lengths,speed_mm_s=speed_mm_s,
        known_motion_time_s=(scan+connector)*1000/speed_mm_s,
        cross_chain_time_s=None if len(lengths)>1 else 0.,
        complete_route_time_s=None if len(lengths)>1 else (scan+connector)*1000/speed_mm_s,
        time_semantics='constant_speed_scan_and_connector_chords_only_excludes_user_pauses_and_unplanned_transitions')


class MotionPlayer:
    def __init__(self,plan,speed=20.):
        self.plan=plan;self.speed=float(speed);statistics(plan,self.speed)
        self.xyz=[np.asarray([p['xyz_m'] for p in c],float) for c in plan['segments']]
        self.uv=[np.asarray([p['pixel'] for p in c],float) for c in plan['segments']]
        self.distances=[np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))] for xyz in self.xyz]
        if not self.xyz or any(len(x)<2 for x in self.xyz):raise ValueError('No playable chain')
        self.offsets=np.r_[0.,np.cumsum([d[-1] for d in self.distances])]
        self.reset()

    def reset(self):
        self.chain=0;self.distance=0.;self.running=False;self.at_break=False;self.finished=False

    def pause(self):self.running=False

    def set_speed(self,value):
        statistics(self.plan,value);self.pause();self.speed=float(value)

    def seek(self,fraction):
        if not np.isfinite(fraction):raise ValueError('Invalid progress')
        target=float(np.clip(fraction,0,1))*self.offsets[-1]
        self.chain=min(len(self.xyz)-1,max(0,int(np.searchsorted(self.offsets,target,side='right'))-1))
        self.distance=min(self.distances[self.chain][-1],target-self.offsets[self.chain])
        self.running=False;self.at_break=False;self.finished=fraction>=1
        return self.position()

    def select_chain(self,index):
        if not 0<=index<len(self.xyz):raise ValueError('Invalid chain')
        self.chain=index;self.distance=0.;self.running=False;self.at_break=False;self.finished=False

    def play(self):
        switched=self.at_break
        if self.finished:self.reset()
        if self.at_break:
            self.chain+=1;self.distance=0.;self.at_break=False
        self.running=True
        return switched

    def advance(self,seconds):
        if not np.isfinite(seconds) or seconds<0:raise ValueError('Invalid time step')
        if self.running:
            end=float(self.distances[self.chain][-1]);self.distance=min(end,self.distance+seconds*self.speed/1000)
            if self.distance>=end:
                self.running=False;self.at_break=self.chain<len(self.xyz)-1;self.finished=not self.at_break
        return self.position()

    def position(self):
        d=self.distances[self.chain];xyz=self.xyz[self.chain];uv=self.uv[self.chain]
        edge=min(len(d)-2,max(0,int(np.searchsorted(d,self.distance,side='right'))-1))
        length=d[edge+1]-d[edge];t=float(np.clip((self.distance-d[edge])/max(length,1e-15),0,1))
        direction=(xyz[edge+1]-xyz[edge])/max(length,1e-15)
        travelled=float(self.offsets[self.chain]+self.distance)
        return dict(chain_index=self.chain,edge_index=edge,xyz_m=(xyz[edge]*(1-t)+xyz[edge+1]*t).tolist(),
            pixel=(uv[edge]*(1-t)+uv[edge+1]*t).tolist(),direction_W0=direction.tolist(),
            motion_kind=self.plan['segments'][self.chain][edge+1].get('motion_kind','scan'),
            progress=travelled/max(self.offsets[-1],1e-15),known_time_s=travelled*1000/self.speed,
            at_break=self.at_break,finished=self.finished)
