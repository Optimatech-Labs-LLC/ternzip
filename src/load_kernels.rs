//! Convert decoded ternary weights directly into the two Prism execution layouts.
#![no_std]
use core::slice;
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    extern "C" { fn abort() -> !; }
    unsafe { abort() }
}

#[no_mangle]
pub unsafe extern "C" fn pack_native(trits:*const u8, scales:*const u16, groups:usize,
                                     stride:usize, output:*mut u8)->i32 {
    if stride != 34 && stride != 28 { return -1; }
    let weights=slice::from_raw_parts(trits,groups*128);
    let scales=slice::from_raw_parts(scales,groups);
    let out=slice::from_raw_parts_mut(output,groups*stride);
    for ((w,block),scale) in weights.chunks_exact(128).zip(out.chunks_exact_mut(stride)).zip(scales) {
        if stride==34 {
            block[..2].copy_from_slice(&scale.to_le_bytes());
            for (dst,s) in block[2..].iter_mut().zip(w.chunks_exact(4)) {
                *dst=s[0] | (s[1]<<2) | (s[2]<<4) | (s[3]<<6);
            }
        } else {
            for &(start,lanes,digits,at) in &[(0,16,5,0),(80,8,5,16),(120,2,4,24)] {
                for j in 0..lanes {
                    let mut v=0u16;
                    for k in 0..digits { v+=u16::from(w[start+k*lanes+j])*[81,27,9,3,1][k]; }
                    block[at+j]=((u32::from(v)*256+242)/243) as u8;
                }
            }
            block[26..28].copy_from_slice(&scale.to_le_bytes());
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn corrected_intervals(keys:*const u16,rows:usize,width:usize,
    base:*const u16,contexts:usize,zero_lut:*const u16,sign_lut:*const u16,
    row_tags:*const i8,col_tags:*const i8,output:*mut u16)->i32 {
    let keys=slice::from_raw_parts(keys,rows*width);
    let base=slice::from_raw_parts(base,contexts*2);
    let zeros=slice::from_raw_parts(zero_lut,contexts*129);
    let signs=slice::from_raw_parts(sign_lut,contexts*129);
    let r=slice::from_raw_parts(row_tags,rows*2);
    let c=slice::from_raw_parts(col_tags,width*2);
    let out=slice::from_raw_parts_mut(output,rows*width*2);
    for ((row,krow),target) in r.chunks_exact(2).zip(keys.chunks_exact(width)).zip(out.chunks_exact_mut(2*width)) {
        for ((&key,col),bounds) in krow.iter().zip(c.chunks_exact(2)).zip(target.chunks_exact_mut(2)) {
            let k=usize::from(key);
            if k>=contexts { return -1; }
            let ztag=(i16::from(row[0])+i16::from(col[0])).clamp(-64,64);
            let stag=(i16::from(row[1])+i16::from(col[1])).clamp(-64,64);
            let zero=zeros[k*129+(ztag+64) as usize];
            let high=if stag==0 && zero==base[2*k] {
                base[2*k+1]
            } else {
                let fraction=u32::from(signs[k*129+(stag+64) as usize]);
                let pos=((u32::from(4096-zero)*fraction+32768)>>16) as u16;
                4096-pos.clamp(1,4095-zero)
            };
            bounds[0]=zero;
            bounds[1]=high;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn row_intervals(reference:*const u8,n:usize,base:*const u16,
                                       odds:*const u16,output:*mut u16)->i32 {
    let r=slice::from_raw_parts(reference,n);
    let b=slice::from_raw_parts(base,n*2);
    let odds=slice::from_raw_parts(odds,9);
    let dst=slice::from_raw_parts_mut(output,n*2);
    for ((&s,prior),out) in r.iter().zip(b.chunks_exact(2)).zip(dst.chunks_exact_mut(2)) {
        if s>2 {return -1;}
        let i=3*usize::from(s);
        let f=&odds[i..i+3];
        if f[0]==f[1] && f[1]==f[2] { out.copy_from_slice(prior); continue; }
        let z=u64::from(prior[0])*u64::from(f[1]);
        let nz=z+u64::from(prior[1]-prior[0])*u64::from(f[0]);
        let sum=nz+u64::from(4096-prior[1])*u64::from(f[2]);
        if sum==0 {return -1;}
        out[0]=(1+(4093*z+sum/2)/sum) as u16;
        out[1]=(2+(4093*nz+sum/2)/sum) as u16;
    }
    0
}

const RANS_LOWER:u32=1<<23;

// Apply only the selected rectangular matches. The source row/column address
// and its sign stay constant for the complete inner word.
#[no_mangle]
pub unsafe extern "C" fn block_intervals(
    previous: *const u8, rows: usize, cols: usize, start_row: usize, count: usize,
    height: usize, word: usize, vertical: u8, global: u8, codes: *const i32,
    mapping: *const i32, flipped: *const u8, site: *const u16, odds: *const u16,
    keys: *const u16, lookup: *const u16, contexts: usize, output: *mut u16,
) -> i32 {
    if height==0 || word==0 || rows%height!=0 || cols%word!=0
        || start_row>rows || count>rows-start_row || vertical>1 || global>1
        || mapping.is_null()!=flipped.is_null() || keys.is_null()!=lookup.is_null() { return -1; }
    let stripes=cols/word;
    let previous=slice::from_raw_parts(previous,rows*cols);
    let codes=slice::from_raw_parts(codes,(rows/height)*stripes);
    let field=slice::from_raw_parts(site,2*count*cols);
    let dst=slice::from_raw_parts_mut(output,2*count*cols);
    let factors=slice::from_raw_parts(odds,9);
    for r in 0..count {
        let row=start_row+r;
        let anchor=if mapping.is_null(){row}else{*mapping.add(row) as usize};
        if anchor>=rows {return -1;}
        let anchor_flip=if flipped.is_null(){0}else{*flipped.add(row)};
        for col in 0..stripes {
            let code=codes[(row/height)*stripes+col];
            if code<0 {continue;}
            let id=code as usize/2;
            let (source_row,source_col,sign)=match (global,vertical) {
                (1,0)=>(id,col,code as u8 & 1),
                (1,1)=>(anchor,id,anchor_flip^(code as u8 & 1)),
                _=>(anchor,col,anchor_flip^(code as u8 & 1)),
            };
            if source_row>=rows || source_col>=stripes {return -1;}
            let start=(r*stripes+col)*word;
            let source=(source_row*stripes+source_col)*word;
            for j in 0..word {
                let p=start+j;
                let value=previous[source+j] as usize;
                if value>2 {return -1;}
                let trit=if sign!=0 {2-value}else{value};
                if !lookup.is_null() {
                    let k=usize::from(*keys.add(p));
                    if k>=contexts {return -1;}
                    let at=2*(trit*contexts+k);
                    dst[2*p]=*lookup.add(at);
                    dst[2*p+1]=*lookup.add(at+1);
                } else {
                    let f=&factors[3*trit..3*trit+3];
                    let z=u64::from(field[2*p]);
                    let n=u64::from(field[2*p+1]);
                    if n<z || n>4096 {return -1;}
                    if f[0]==f[1] && f[1]==f[2] {
                        dst[2*p]=z as u16;
                        dst[2*p+1]=n as u16;
                        continue;
                    }
                    let zero=z*u64::from(f[1]);
                    let nonpos=zero+(n-z)*u64::from(f[0]);
                    let sum=nonpos+(4096-n)*u64::from(f[2]);
                    if sum==0 {return -1;}
                    dst[2*p]=(1+(4093*zero+sum/2)/sum) as u16;
                    dst[2*p+1]=(2+(4093*nonpos+sum/2)/sum) as u16;
                }
            }
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn inverse_embedding(row:*mut f32,width:usize,signs:*const f32,scale:f32)->i32 {
    if width==0 || !width.is_power_of_two() {return -1;}
    let row=slice::from_raw_parts_mut(row,width);
    let signs=slice::from_raw_parts(signs,width);
    let mut half=1;
    while half<width {
        for block in row.chunks_exact_mut(half*2) {
            let (left,right)=block.split_at_mut(half);
            for (a,b) in left.iter_mut().zip(right.iter_mut()) {
                let x=*a;
                let y=*b;
                *a=x+y;
                *b=x-y;
            }
        }
        half*=2;
    }
    for (value,sign) in row.iter_mut().zip(signs) { *value *= scale*sign; }
    0
}
fn walk(states:&mut [u32;4], bytes:&[u8], pos:&mut usize, keys:&[u16], table:&[u16],
        mut output:Option<&mut [u8]>)->bool {
    for (i,key) in keys.iter().enumerate() {
        let k=2*usize::from(*key);
        if k+1>=table.len() {return false;}
        let z=u32::from(table[k]); let h=u32::from(table[k+1]);
        if z==0 || h<=z || h>=4096 {return false;}
        let mut x=states[i&3]; let slot=x&4095;
        let (trit,start,freq)=if slot<z {(1,0,z)} else if slot<h {(0,z,h-z)} else {(2,h,4096-h)};
        if let Some(ref mut out)=output {out[i]=trit;}
        x=freq*(x>>12)+slot-start;
        while x<RANS_LOWER {
            if *pos==bytes.len() {return false;}
            x=(x<<8)|u32::from(bytes[*pos]); *pos+=1;
        }
        states[i&3]=x;
    }
    true
}

#[no_mangle]
pub unsafe extern "C" fn index_stream(payload:*const u8,len:usize,keys:*const u16,rows:usize,width:usize,
                                      bounds:*const u16,contexts:usize,output:*mut u32)->i32 {
    if len<16 || len>u32::MAX as usize || width%4!=0 || width==0{return -1;}
    let bytes=slice::from_raw_parts(payload,len);
    let keys=slice::from_raw_parts(keys,rows*width);
    let table=slice::from_raw_parts(bounds,2*contexts);
    let out=slice::from_raw_parts_mut(output,(rows+1)*5);
    let mut states=[0u32;4];let mut pos=16;
    for (j,s) in states.iter_mut().enumerate() {
        *s=u32::from_le_bytes(bytes[4*j..4*j+4].try_into().unwrap());
        if !(RANS_LOWER..RANS_LOWER<<8).contains(s){return -1;}
    }
    for (j,row) in keys.chunks_exact(width).enumerate() {
        out[5*j]=pos as u32;out[5*j+1..5*j+5].copy_from_slice(&states);
        if !walk(&mut states,bytes,&mut pos,row,table,None){return -1;}
    }
    out[5*rows]=pos as u32;out[5*rows+1..].copy_from_slice(&states);
    if pos!=len || states!=[RANS_LOWER;4] {return -1;}
    0
}

#[no_mangle]
pub unsafe extern "C" fn seek_row(payload:*const u8,len:usize,check:*const u32,
                                  keys:*const u16,width:usize,bounds:*const u16,contexts:usize,out:*mut u8)->i32 {
    let bytes=slice::from_raw_parts(payload,len); let check=slice::from_raw_parts(check,10);
    let mut states:[u32;4]=check[1..5].try_into().unwrap();let mut pos=0;
    if check[5].checked_sub(check[0])!=Some(len as u32) {return -1;}
    if !walk(&mut states,bytes,&mut pos,slice::from_raw_parts(keys,width),slice::from_raw_parts(bounds,2*contexts),
             Some(slice::from_raw_parts_mut(out,width))) {return -1;}
    if pos!=len || states[..]!=check[6..10] {return -1;}
    0
}
