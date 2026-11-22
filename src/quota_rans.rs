//! Four-way rANS with a conditional zero count in each block of 128 trits.
#![no_std]
use core::slice;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    extern "C" { fn abort() -> !; }
    unsafe { abort() }
}

const UNIT: u32 = 4096;
const LOWER: u32 = 1 << 23;
const PLANE: usize = 129 * 129;

#[inline(always)]
fn intervals(models: &[u32], key: usize, grid: &[u16], slots: usize, zeros: usize) -> (u32, u32) {
    let base = models[key * 2] as usize;
    let z = u32::from(grid[base + slots * 129 + zeros]);
    if z == UNIT { return (UNIT, UNIT); }
    let space = UNIT - z;
    let neg = ((space * models[key * 2 + 1] + 32768) >> 16).clamp(1, space - 1);
    (z, z + neg)
}

#[no_mangle]
pub unsafe extern "C" fn quota_encode(
    trits: *const u8, keys: *const u16, n: usize, models: *const u32, contexts: usize,
    grid: *const u16, counts: *mut u8, out: *mut u8, capacity: usize,
) -> isize {
    if n % 128 != 0 || capacity < 16 { return -1; }
    let symbols = slice::from_raw_parts(trits, n);
    let keys = slice::from_raw_parts(keys, n);
    let models = slice::from_raw_parts(models, 2 * contexts);
    let grid = slice::from_raw_parts(grid, 65 * PLANE);
    let counts = slice::from_raw_parts_mut(counts, n / 128);
    let out = slice::from_raw_parts_mut(out, capacity);
    let mut states = [LOWER; 4];
    let mut zeros = 0usize;
    let mut at = capacity;
    for i in (0..n).rev() {
        let key = keys[i] as usize;
        if key >= contexts { return -1; }
        let digit = symbols[i];
        zeros += usize::from(digit == 1);
        let (z, neg) = intervals(models, key, grid, 128 - (i & 127), zeros);
        let (start, end) = match digit { 0 => (z, neg), 1 => (0,z), 2 => (neg,UNIT), _ => return -1 };
        let freq = end - start;
        if freq == 0 { return -1; }
        let mut x = states[i & 3];
        let limit = ((LOWER / UNIT) << 8) * freq;
        while x >= limit {
            if at <= 16 { return -1; }
            at -= 1; out[at] = x as u8; x >>= 8;
        }
        states[i & 3] = (x / freq) * UNIT + x % freq + start;
        if i & 127 == 0 { counts[i / 128] = zeros as u8; zeros = 0; }
    }
    at -= 16;
    for (lane, state) in states.iter().enumerate() {
        out[at + 4 * lane..at + 4 * lane + 4].copy_from_slice(&state.to_le_bytes());
    }
    at as isize
}

#[no_mangle]
pub unsafe extern "C" fn quota_decode(
    payload: *const u8, length: usize, keys: *const u16, n: usize, models: *const u32,
    contexts: usize, grid: *const u16, counts: *const u8, out: *mut u8,
) -> i32 {
    if length < 16 || n % 128 != 0 { return -1; }
    let data = slice::from_raw_parts(payload, length);
    let keys = slice::from_raw_parts(keys, n);
    let models = slice::from_raw_parts(models, 2 * contexts);
    let grid = slice::from_raw_parts(grid, 65 * PLANE);
    let counts = slice::from_raw_parts(counts, n / 128);
    let out = slice::from_raw_parts_mut(out, n);
    let mut states = [0u32; 4];
    for (lane, state) in states.iter_mut().enumerate() {
        *state = u32::from_le_bytes(data[4*lane..4*lane+4].try_into().unwrap());
        if !(LOWER..LOWER << 8).contains(state) { return -1; }
    }
    let mut at = 16;
    let mut zeros = 0usize;
    for i in 0..n {
        if i & 127 == 0 { zeros = counts[i / 128] as usize; }
        let slots = 128 - (i & 127);
        let key = keys[i] as usize;
        if key >= contexts || zeros > slots { return -1; }
        let (z, neg) = intervals(models, key, grid, slots, zeros);
        let mut x = states[i & 3];
        let slot = x & (UNIT - 1);
        let (symbol, start, freq) = if slot < z { zeros -= 1; (1, 0, z) }
            else if slot < neg { (0, z, neg-z) } else { (2, neg, UNIT-neg) };
        out[i] = symbol;
        x = freq * (x / UNIT) + slot - start;
        while x < LOWER {
            if at >= length { return -1; }
            x = (x << 8) | u32::from(data[at]); at += 1;
        }
        states[i & 3] = x;
    }
    if at != length || zeros != 0 || states != [LOWER; 4] { return -1; }
    0
}
