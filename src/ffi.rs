use std::ffi::c_char;
use crate::chess::ChessModel;

#[unsafe(no_mangle)]
pub extern "C" fn chess_model_load(path: *const c_char) -> *mut ChessModel {
    let c_str = unsafe { std::ffi::CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match ChessModel::load_from_file(path_str) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn chess_model_free(model: *mut ChessModel) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn chess_model_evaluate(
    model: *mut ChessModel,
    input_ptr: *const u16,
    input_len: usize,
    legality_mask_ptr: *const bool,
    mask_len: usize,
    out_policy: *mut f32,
    out_value_wdl: *mut f32, // Array of size 3 (Win, Draw, Loss)
) {
    let model = unsafe { &mut *model };
    let input = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    let legality_mask = unsafe { std::slice::from_raw_parts(legality_mask_ptr, mask_len) };

    let output = model.step(input, legality_mask, None);

    unsafe {
        std::ptr::copy_nonoverlapping(output.policy.as_ptr(), out_policy, output.policy.len());
        std::ptr::copy_nonoverlapping(output.value.as_ptr(), out_value_wdl, 3);
    }
}
