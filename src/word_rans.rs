//! A 30-bit rANS symbol describes two consecutive four-error words.
//! Each interval starts with the product of the eight position-specific trit probabilities.
#![no_std]
use core::slice;
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    extern "C" { fn abort() -> !; }
    unsafe { abort() }
}

const TOTAL: u64 = 1 << 30;
const LOWER: u64 = 1 << 40;
const WORDS: usize = 81 * 81;
const A: usize = 2*81+72; // ---+ followed by ++--
const LO: [usize;4] = [0,2,39,72];
const HI: [usize;4] = [2,3,42,73];

struct Word {
    freq: [[f64;3];4],
    pred: [u8;4],
    lower: [f64;4],
    gain: [f64;4],
    full: [f64;4],
    norm: f64,
    neutral: bool,
}
impl Word {
    fn new(bounds: &[u16], factors: &[u16]) -> Option<Self> {
        let mut w=Self {freq:[[0.;3];4], pred:[0;4], lower:[0.;4],
                        gain:[0.;4],full:[0.;4],norm:1.,neutral:factors[..4]==[256;4]};
        for j in 0..4 {
            let z=bounds[2*j];
            let n=bounds[2*j+1];
            if z==0 || n<=z || n>=4096 { return None; }
            let f=[n-z,z,4096-n];
            let mut pred=0;
            for s in 1..3 { if f[s]>f[pred] { pred=s; } }
            w.pred[j]=pred as u8;
            for r in 0..3 { w.freq[j][r]=f[(r+pred+2)%3] as f64 /4096.; }
        }
        if w.neutral { return Some(w); }
        for t in 0..4 {
            w.lower[t]=w.raw(LO[t]);
            w.gain[t]=factors[t] as f64/256.-1.;
            w.full[t]=(w.raw(HI[t])-w.lower[t])*w.gain[t];
            w.norm+=w.full[t];
        }
        Some(w)
    }
    #[inline]
    fn raw(&self,mut code:usize)->f64 {
        if code>=81 {return 1.;}
        let mut lo=0.;
        let mut mass=1.;
        for (i, place) in [27,9,3,1].iter().enumerate() {
            let d=code/place;
            code%=place;
            let f=self.freq[i];
            if d>0 { lo+=mass*if d==1 { f[0] } else { f[0]+f[1] }; }
            mass*=f[d];
        }
        lo
    }
    #[inline]
    fn edge(&self, code:usize)->f64 {
        if code==0 {return 0.;}
        if code>=81 {return 1.;}
        let raw=self.raw(code);
        if self.neutral { return raw; }
        let mut lo=raw;
        for t in 0..4 {
            if code>=HI[t] {lo+=self.full[t];}
            else if code>LO[t] {lo+=(raw-self.lower[t])*self.gain[t];}
        }
        lo/self.norm
    }
    // Invert an unadjusted four-trit distribution. The coder separately checks
    // the integer endpoints, which include one reserved slot per possible word.
    fn quantile(&self, mut fraction:f64)->usize {
        let mut rank=0;
        for f in self.freq {
            let (digit,lower)=if fraction<f[0] {
                (0,0.)
            } else if fraction<f[0]+f[1] {
                (1,f[0])
            } else {
                (2,f[0]+f[1])
            };
            rank=rank*3+digit;
            fraction=(fraction-lower)/f[digit];
        }
        rank
    }
    fn expand(&self,mut id:usize,out:&mut[u8]) {
        for i in (0..4).rev() {
            out[i]=((id%3) as u8+self.pred[i]+2)%3;
            id/=3;
        }
    }
}

struct Pair {first:Word,second:Word,delta:f64}
impl Pair {
    fn new(bounds:&[u16],factors:&[u16])->Option<Self> {
        let first=Word::new(&bounds[..8],factors)?;
        let second=Word::new(&bounds[8..],factors)?;
        let delta=if factors[4]==256 {0.} else {
            (first.edge(3)-first.edge(2))*(second.edge(73)-second.edge(72))*(factors[4] as f64/256.-1.)
        };
        Some(Self {first,second,delta})
    }
    #[inline]
    fn fixed(&self,rank:usize,prob:f64)->u64 {
        if rank==WORDS {return TOTAL;}
        let q=(prob+if rank>A {self.delta} else {0.})/(1.+self.delta);
        rank as u64+((TOTAL-WORDS as u64) as f64*q.clamp(0.,1.)) as u64
    }
    fn span(&self,a:usize,b:usize)->(u64,u64) {
        let lo=self.first.edge(a);
        let next=self.first.edge(a+1);
        let rank=81*a+b;
        let low=self.fixed(rank,lo+(next-lo)*self.second.edge(b));
        let upper=if b==80 {next} else {lo+(next-lo)*self.second.edge(b+1)};
        (low,self.fixed(rank+1,upper)-low)
    }
    fn symbol(&self,slot:u64)->(usize,usize) {
        let mut low=0; let mut high=81;
        while low+1<high {
            let mid=(low+high)/2;
            if self.fixed(mid*81,self.first.edge(mid))<=slot {low=mid;} else {high=mid;}
        }
        let a=low;
        let lo=self.first.edge(a); let mass=self.first.edge(a+1)-lo;
        low=0; high=81;
        while low+1<high {
            let mid=(low+high)/2;
            if self.fixed(a*81+mid,lo+mass*self.second.edge(mid))<=slot {low=mid;} else {high=mid;}
        }
        (a,low)
    }
    fn decode_span(&self,slot:u64)->(usize,usize,u64,u64) {
        if self.first.neutral && self.delta==0. {
            let fraction=slot as f64/TOTAL as f64;
            let a=self.first.quantile(fraction);
            let lo=self.first.raw(a);
            let mass=self.first.raw(a+1)-lo;
            if mass>0. {
                let b=self.second.quantile((fraction-lo)/mass);
                let (start,freq)=self.span(a,b);
                if start<=slot && slot<start+freq { return (a,b,start,freq); }
            }
        }
        let (a,b)=self.symbol(slot);
        let (start,freq)=self.span(a,b);
        (a,b,start,freq)
    }
}

#[no_mangle]
pub unsafe extern "C" fn word_decode(payload:*const u8,bytes:usize,bounds:*const u16,factors:*const u16,
                                     out:*mut u8,n:usize)->i32 {
    if bytes<8 || n%8!=0 {return -1;}
    let p=slice::from_raw_parts(payload,bytes); let bounds=slice::from_raw_parts(bounds,2*n);
    let factors=slice::from_raw_parts(factors,5); let out=slice::from_raw_parts_mut(out,n);
    if factors.iter().any(|&v| v==0) {return -1;}
    let mut x=u64::from_le_bytes(p[..8].try_into().unwrap());
    if !(LOWER..LOWER<<8).contains(&x) {return -1;}
    let mut at=8;
    for i in 0..n/8 {
        let Some(pair)=Pair::new(&bounds[16*i..16*i+16],factors) else{return -1;};
        let slot=x&(TOTAL-1); let (a,b,start,freq)=pair.decode_span(slot);
        if slot<start || slot>=start+freq {return -1;}
        pair.first.expand(a,&mut out[8*i..8*i+4]);
        pair.second.expand(b,&mut out[8*i+4..8*i+8]);
        x=freq*(x/TOTAL)+slot-start;
        while x<LOWER {
            if at==bytes {return -1;} x=(x<<8)|p[at] as u64; at+=1;
        }
    }
    if at!=bytes || x!=LOWER {return -1;} 0
}
