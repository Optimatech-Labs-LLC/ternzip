//! Four interleaved byte-rANS states. The two bounds describe zero and sign.
#![no_std]

use core::slice;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    extern "C" {
        fn abort() -> !;
    }
    unsafe { abort() }
}

const UNIT: u32 = 4096;
const LOWER: u32 = 1 << 23;

#[no_mangle]
pub unsafe extern "C" fn trit_encode(
    trits: *const u8,
    keys: *const u16,
    n: usize,
    bounds: *const u16,
    contexts: usize,
    out: *mut u8,
    capacity: usize,
) -> isize {
    let symbols = slice::from_raw_parts(trits, n);
    let keys = if keys.is_null() { &[] } else { slice::from_raw_parts(keys,n) };
    let bounds = slice::from_raw_parts(bounds, 2 * contexts);
    let out = slice::from_raw_parts_mut(out, capacity);
    let mut states = [LOWER; 4];
    let mut at = capacity;
    for i in (0..n).rev() {
        let k = if keys.is_empty() { i } else { usize::from(keys[i]) };
        if k >= contexts {
            return -1;
        }
        let zero = u32::from(bounds[2 * k]);
        let neg = u32::from(bounds[2 * k + 1]);
        if zero == 0 || neg <= zero || neg >= UNIT {
            return -1;
        }
        let (start, end) = match symbols[i] {
            1 => (0, zero),
            0 => (zero, neg),
            2 => (neg, UNIT),
            _ => return -1,
        };
        let freq = end - start;
        let limit = ((LOWER / UNIT) << 8) * freq;
        let mut x = states[i & 3];
        while x >= limit {
            if at <= 16 {
                return -1;
            }
            at -= 1;
            out[at] = x as u8;
            x >>= 8;
        }
        states[i & 3] = (x / freq) * UNIT + x % freq + start;
    }
    if at < 16 {
        return -1;
    }
    at -= 16;
    for (lane, state) in states.iter().enumerate() {
        for (byte, v) in state.to_le_bytes().iter().enumerate() {
            out[at + 4 * lane + byte] = *v;
        }
    }
    at as isize
}

#[no_mangle]
pub unsafe extern "C" fn trit_decode(
    payload: *const u8,
    length: usize,
    keys: *const u16,
    n: usize,
    bounds: *const u16,
    contexts: usize,
    out: *mut u8,
) -> i32 {
    if length < 16 {
        return -1;
    }
    let payload = slice::from_raw_parts(payload, length);
    let keys = if keys.is_null() { &[] } else { slice::from_raw_parts(keys,n) };
    let bounds = slice::from_raw_parts(bounds, 2 * contexts);
    let out = slice::from_raw_parts_mut(out, n);
    let mut states = [0u32; 4];
    for (lane, state) in states.iter_mut().enumerate() {
        let p = 4 * lane;
        *state = u32::from_le_bytes([payload[p], payload[p+1], payload[p+2], payload[p+3]]);
        if !(LOWER..(LOWER << 8)).contains(state) {
            return -1;
        }
    }
    let mut at = 16;
    for i in 0..n {
        let k = if keys.is_empty() { i } else { usize::from(keys[i]) };
        if k >= contexts {
            return -1;
        }
        let zero = u32::from(bounds[2 * k]);
        let neg = u32::from(bounds[2 * k + 1]);
        if zero == 0 || neg <= zero || neg >= UNIT {
            return -1;
        }
        let mut x = states[i & 3];
        let slot = x & (UNIT - 1);
        let (symbol, start, freq) = if slot < zero {
            (1, 0, zero)
        } else if slot < neg {
            (0, zero, neg - zero)
        } else {
            (2, neg, UNIT - neg)
        };
        out[i] = symbol;
        x = freq * (x / UNIT) + slot - start;
        while x < LOWER {
            if at >= length {
                return -1;
            }
            x = (x << 8) | u32::from(payload[at]);
            at += 1;
        }
        states[i & 3] = x;
    }
    if at != length || states.iter().any(|&x| x != LOWER) {
        return -1;
    }
    0
}
