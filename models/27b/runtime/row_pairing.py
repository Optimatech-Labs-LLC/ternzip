import numpy as np
FIXED_DIGITS=((np.arange(256,dtype=np.uint32)*243//256)[:,None]//np.array([81,27,9,3,1])%3).astype(np.uint8)


def unpack_ptq(blocks):
    trits=np.empty((len(blocks),128),np.uint8)
    for begin,lanes,digits,at in ((0,16,5,0),(80,8,5,16),(120,2,4,24)):
        dst=trits[:,begin:begin+lanes*digits].reshape(-1,digits,lanes)
        dst[:]=FIXED_DIGITS[blocks[:,at:at+lanes],:digits].swapaxes(1,2)
    return trits

