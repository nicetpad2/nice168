# ==============================================================================
# === PART 1/15: Setup, Imports, Global Variable Loading, Basic Fixtures ===
# ==============================================================================
# <<< MODIFIED: [Patch - IMPORT ERROR FIX - Step 1-5 Combined] >>>
# <<< Integrating changes for pynvml import, library installation logging, Colab detection debugging,
#     robust handling of optional library imports, and deferring GPU setup and FileHandler from global scope. >>>

import os
import sys
import subprocess
import importlib
import logging
import logging.handlers # Keep for potential use in main()
import datetime
import time
import warnings
import traceback
# import pandas as pd # Deferred: Will be imported robustly or via try_import_with_install
# import numpy as np  # Deferred: Will be imported robustly or via try_import_with_install
import json
import gzip
import gc
from collections import defaultdict
from typing import Union, Optional, Callable, Any, Dict, List, Tuple

# --- Script Version and Basic Setup ---
MINIMAL_SCRIPT_VERSION = "4.9.25_AISTUDIO_PATCH_IMPORT_FIX_V2" # Updated version

# --- Global Variables for Library Availability ---
tqdm_imported = False
ta_imported = False
optuna_imported = False
catboost_imported = False
shap_imported = False
gputil_imported = False
pynvml = None # Initialized to None, will be set by setup_gpu_acceleration if successful
# pynvml_imported = False # Optional flag, manage if used
psutil_imported = False
torch_imported = False
pandas_imported = False
numpy_imported = False

# --- Global Variables for System State ---
IN_COLAB = False
USE_GPU_ACCELERATION = True # Default, will be checked by setup_gpu_acceleration (called from main)
nvml_handle = None # For pynvml GPU monitoring, set by setup_gpu_acceleration
OUTPUT_DIR: str = "./gold_ai_output_temp_import_v2" # Temporary default, main() should override
LOG_FILENAME: str = f"gold_ai_v{MINIMAL_SCRIPT_VERSION.split('_')[0]}_temp_import_v2.log"

# --- Logger Setup ---
logger = logging.getLogger("GoldAI_Enterprise_v4.9")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    logger.info(f"[Patch - IMPORT ERROR FIX - Step 2 & 3] Initial StreamHandler for logger configured. FileHandler deferred to main().")

logger.info(f"Gold AI Script Version: {MINIMAL_SCRIPT_VERSION} - Logger Initialized (Import Phase).")
logger.info("[Patch - IMPORT ERROR FIX - Step 1 (Manual)] Developer to review entire gold_ai2025.py for syntax errors.")
logger.info("[Patch - IMPORT ERROR FIX - Step 2 (Review)] Reviewing top-level imports and global scope code in gold_ai2025.py.")

# --- Helper Function to Log Library Versions ---
def log_library_version(library_name: str, library_module: Optional[Any]):
    log_ver_logger = logging.getLogger(f"{__name__}.log_library_version")
    if library_module is None:
        log_ver_logger.warning(f"Library {library_name.upper()} is None (likely not imported or import failed).")
        return
    try:
        version = getattr(library_module, '__version__', None)
        if version and version != 'N/A':
            log_ver_logger.info(f"(Info) Using {library_name.upper()} version: {version}")
        else:
            log_ver_logger.debug(f"[DEBUG] Version attribute for library {library_name.upper()} is 'N/A' or None.")
    except AttributeError:
        log_ver_logger.debug(f"[DEBUG] Version attribute for library {library_name.upper()} not found (AttributeError).")
    except Exception as e: # pragma: no cover
        log_ver_logger.warning(f"[DEBUG] Could not retrieve version for {library_name.upper()}: {e}")

# --- Helper Function for Conditional Library Import and Installation ---
def try_import_with_install(
    module_name: str,
    pip_install_name: Optional[str] = None,
    import_as_name: Optional[str] = None,
    success_flag_global_name: Optional[str] = None,
    log_name: Optional[str] = None
) -> Optional[Any]:
    import_helper_logger = logging.getLogger(f"{__name__}.try_import_with_install")
    actual_log_name = log_name if log_name else module_name.upper()
    actual_pip_name = pip_install_name if pip_install_name else module_name

    try:
        imported_module = importlib.import_module(module_name)
        if import_as_name:
            globals()[import_as_name] = imported_module
        if success_flag_global_name:
            globals()[success_flag_global_name] = True
        log_library_version(actual_log_name, imported_module)
        return imported_module
    except ImportError:
        import_helper_logger.warning(f"ไม่พบ Library: {actual_log_name}. กำลังพยายามติดตั้ง...")
        if pip_install_name:
            try:
                install_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", actual_pip_name, "-q"],
                    check=True, capture_output=True, text=True
                )
                import_helper_logger.info(f"   ติดตั้ง {actual_log_name} สำเร็จ (pip stdout): {install_result.stdout[:100]}...")
                imported_module_after_install = importlib.import_module(module_name)
                if import_as_name:
                    globals()[import_as_name] = imported_module_after_install
                if success_flag_global_name:
                    globals()[success_flag_global_name] = True
                
                # [Patch AI Studio - Log 16:57:37 - Part 2 - Combined with Patch G]
                version_after_install = getattr(imported_module_after_install, '__version__', 'N/A')
                if version_after_install != 'N/A':
                    import_helper_logger.info(f"[Patch] Successfully installed and imported {actual_log_name} version: {version_after_install}")
                else:
                    import_helper_logger.info(f"[Patch] Successfully installed and imported {actual_log_name}.")
                # End of Patch
                return imported_module_after_install
            except subprocess.CalledProcessError as e_install: # pragma: no cover
                import_helper_logger.error(f"   (Error) ไม่สามารถติดตั้ง {actual_log_name} (pip error): {e_install.stderr[:200]}...")
            except ImportError: # pragma: no cover
                import_helper_logger.error(f"   (Error) ไม่สามารถ Import {actual_log_name} ได้ แม้หลังจากการติดตั้ง.")
            except Exception as e_generic_install: # pragma: no cover
                 import_helper_logger.error(f"   (Error) เกิดข้อผิดพลาดระหว่างติดตั้ง/Import {actual_log_name}: {e_generic_install}", exc_info=True)
        else: # pragma: no cover
            import_helper_logger.warning(f"   (Warning) ไม่ได้ระบุชื่อ Pip สำหรับ {actual_log_name}. ไม่สามารถติดตั้งได้.")
    except Exception as e_outer_import: # pragma: no cover
        import_helper_logger.error(f"   (Error) Unexpected error during initial import of {actual_log_name}: {e_outer_import}", exc_info=True)

    # [Patch - IMPORT ERROR FIX - Step 5] Ensure flags are False and dummies assigned if import fails
    if success_flag_global_name and success_flag_global_name in globals(): # Check if flag exists before trying to set
        if not globals()[success_flag_global_name]: # Only if it's still False (meaning import failed)
            globals()[success_flag_global_name] = False # Explicitly set to False
            if import_as_name:
                if import_as_name == "tqdm":
                    globals()[import_as_name] = lambda x, *args, **kwargs: x
                    import_helper_logger.info(f"   Assigned dummy function for missing '{import_as_name}'.")
                else:
                    globals()[import_as_name] = None
                    import_helper_logger.info(f"   Assigned None for missing '{import_as_name}'.")
    return None

# --- Import Core Libraries (More Robustly) ---
logger.info("\n(Processing) Importing core libraries (with robust fallbacks)...")
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format", module='pandas')

# Pandas and NumPy (critical, attempt import, log if fails but script might be unusable)
# These are now imported via try_import_with_install for consistency and to allow pip fallback if needed
# However, for critical libraries like pandas and numpy, it's often better to ensure they are pre-installed.
# The dummy classes are a last resort to make the script *loadable* by pytest, not fully functional.

pd = try_import_with_install("pandas", import_as_name="pd", success_flag_global_name="pandas_imported")
if not pandas_imported: # pragma: no cover
    logger.critical("[CRITICAL IMPORT FAIL] Pandas library could not be imported or installed. Many functionalities will fail.")
    class DummyPandas:
        DataFrame = type('DummyDataFrame', (object,), {})
        Series = type('DummySeries', (object,), {})
        Timestamp = type('DummyTimestamp', (object,), {})
        NaT = None
        DatetimeIndex = type('DummyDatetimeIndex', (object,), {}) # Added
        def to_datetime(self, *args, **kwargs): return None
        def read_csv(self, *args, **kwargs): return self.DataFrame()
        def merge_asof(self, *args, **kwargs): return self.DataFrame()
        def concat(self, *args, **kwargs): return self.DataFrame() # Added
        class errors: # Added
            ParserError = type('DummyParserError', (Exception,), {})
            EmptyDataError = type('DummyEmptyDataError', (Exception,), {})
        class api:
            class types:
                @staticmethod
                def is_datetime64_any_dtype(val): return False
                @staticmethod
                def is_numeric_dtype(val): return isinstance(val, (int, float))
                @staticmethod
                def is_integer_dtype(val): return isinstance(val, int)
                @staticmethod
                def is_float_dtype(val): return isinstance(val, float)
    globals()['pd'] = DummyPandas()

np = try_import_with_install("numpy", import_as_name="np", success_flag_global_name="numpy_imported")
if not numpy_imported: # pragma: no cover
    logger.critical("[CRITICAL IMPORT FAIL] NumPy library could not be imported or installed. Many functionalities will fail.")
    class DummyNumpy:
        nan = float('nan') # Use float('nan') for better compatibility
        inf = float('inf')
        integer = int
        floating = float
        bool_ = bool
        ndarray = list # Treat ndarray as list for dummy
        def array(self, *args, **kwargs): return list(args[0]) if args and args[0] is not None else []
        def mean(self, *args, **kwargs): return 0.0
        def std(self, *args, **kwargs): return 0.0
        def abs(self, val): return abs(val) if isinstance(val, (int, float)) else ([abs(x) for x in val] if isinstance(val, list) else 0)
        def where(self, condition, x, y): return [x_val if c else y_val for c, x_val, y_val in zip(condition, x, y)] # Simplified
        def isinf(self, val): return val == float('inf') or val == float('-inf')
        def isnan(self, val): return val != val # Standard way to check for float NaN
        def maximum(self, *args, **kwargs): return max(args) if args else 0
        def minimum(self, *args, **kwargs): return min(args) if args else 0
        def sign(self, val): return 0 if val == 0 else (1 if val > 0 else -1)
        def clip(self, arr, min_val, max_val): return [max(min_val, min(x, max_val)) for x in arr] if isinstance(arr, list) else max(min_val, min(arr, max_val))
        class errstate:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
    globals()['np'] = DummyNumpy()


# tqdm (with fallback)
tqdm_module = try_import_with_install("tqdm.notebook", pip_install_name="tqdm", import_as_name="tqdm", success_flag_global_name="tqdm_imported", log_name="TQDM.NOTEBOOK")
if not tqdm_imported or tqdm_module is None: # pragma: no cover
    tqdm = lambda x, *args, **kwargs: x 
    logger.warning("   (Fallback) Using dummy tqdm as tqdm.notebook import failed.")
else:
    tqdm = tqdm_module

# ta (Technical Analysis Library)
ta = try_import_with_install("ta", success_flag_global_name="ta_imported")

# Optuna (Hyperparameter Optimization)
optuna = try_import_with_install("optuna", success_flag_global_name="optuna_imported")
if optuna_imported and optuna: # pragma: no cover
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception as e_optuna_log:
        logger.warning(f"   (Warning) Could not set Optuna verbosity: {e_optuna_log}")

# CatBoost
CatBoostClassifier = None
Pool = None
EShapCalcType = None
EFeaturesSelectionAlgorithm = None
catboost_module = try_import_with_install("catboost", success_flag_global_name="catboost_imported")
if catboost_imported and catboost_module: # pragma: no cover
    try:
        CatBoostClassifier = getattr(catboost_module, 'CatBoostClassifier')
        Pool = getattr(catboost_module, 'Pool')
        EShapCalcType = getattr(catboost_module, 'EShapCalcType', None)
        EFeaturesSelectionAlgorithm = getattr(catboost_module, 'EFeaturesSelectionAlgorithm', None)
        logger.info(f"   (Success) CatBoost components (Classifier, Pool) loaded. EShapCalcType: {'Found' if EShapCalcType else 'Not Found'}, EFeaturesSelectionAlgorithm: {'Found' if EFeaturesSelectionAlgorithm else 'Not Found'}")
    except AttributeError as e_cat_attr:
        logger.error(f"   (Error) Could not get CatBoost components from module: {e_cat_attr}")
        catboost_imported = False
        CatBoostClassifier = None; Pool = None; EShapCalcType = None; EFeaturesSelectionAlgorithm = None

# SHAP (Explainable AI)
shap = try_import_with_install("shap", success_flag_global_name="shap_imported")

# GPUtil (GPU Monitoring)
GPUtil = try_import_with_install("GPUtil", import_as_name="GPUtil", success_flag_global_name="gputil_imported")

# psutil (System Utilities)
psutil = try_import_with_install("psutil", success_flag_global_name="psutil_imported")

# PyTorch (Deep Learning, GPU check)
torch = None
try:
    import torch
    torch_imported = True
    log_library_version("PyTorch", torch)
except ImportError: # pragma: no cover
    logger.warning("PyTorch not found. GPU acceleration will be disabled if it was intended.")
    torch_imported = False
except Exception as e_torch_import: # pragma: no cover
    logger.error(f"Error importing PyTorch: {e_torch_import}. GPU acceleration might be affected.")
    torch_imported = False


# --- Environment Setup (Colab, GPU) ---
logger.info("\n(Processing) Setting up environment (Colab, GPU)...")
try:
    from IPython import get_ipython # type: ignore
    shell = get_ipython()
    logger.debug(f"[IN_COLAB Check - Patch G] Value from get_ipython(): {shell}")
    if shell is not None:
        shell_str = str(shell)
        logger.debug(f"[IN_COLAB Check - Patch G] Value from str(get_ipython()): {shell_str}")
        if 'google.colab' in shell_str:
            IN_COLAB = True
            logger.info("   Running in Google Colab environment.")
            try:
                from google.colab import drive # type: ignore
                logger.info("   Attempting to mount Google Drive...")
                drive.mount('/content/drive', force_remount=True)
                logger.info("   Google Drive mounted successfully.")
            except ImportError: # pragma: no cover
                logger.error("   (Error) Failed to import google.colab.drive. Drive mounting skipped.")
                drive = type('DummyDrive', (), {'mount': lambda *args, **kwargs: logger.warning("DummyDrive.mount called.")})() # type: ignore
            except Exception as e_drive: # pragma: no cover
                logger.error(f"   (Error) Failed to mount Google Drive: {e_drive}", exc_info=True)
                drive = type('DummyDrive', (), {'mount': lambda *args, **kwargs: logger.warning("DummyDrive.mount called.")})() # type: ignore
        else: # pragma: no cover
            logger.info("   Not running in Google Colab environment (based on get_ipython string).")
            drive = type('DummyDrive', (), {'mount': lambda *args, **kwargs: logger.warning("DummyDrive.mount called.")})() # type: ignore
    else: # pragma: no cover
        logger.info("   Not running in Google Colab environment (get_ipython is None).")
        drive = type('DummyDrive', (), {'mount': lambda *args, **kwargs: logger.warning("DummyDrive.mount called.")})() # type: ignore
except ImportError: # pragma: no cover
    logger.info("   IPython not found. Assuming not in Colab environment.")
    drive = type('DummyDrive', (), {'mount': lambda *args, **kwargs: logger.warning("DummyDrive.mount called.")})() # type: ignore
except Exception as e_colab_setup: # pragma: no cover
    logger.error(f"   Error during Colab environment setup: {e_colab_setup}", exc_info=True)
    drive = type('DummyDrive', (), {'mount': lambda *args, **kwargs: logger.warning("DummyDrive.mount called.")})() # type: ignore


# --- GPU Acceleration Setup Function Definition (will be called from __main__) ---
def setup_gpu_acceleration():
    global USE_GPU_ACCELERATION, nvml_handle, pynvml
    gpu_setup_logger = logging.getLogger(f"{__name__}.setup_gpu_acceleration")
    gpu_setup_logger.info("   Checking GPU availability for acceleration (called from main)...") # Log context

    if not torch_imported or not hasattr(torch, 'cuda') or not hasattr(torch.cuda, 'is_available'): # pragma: no cover
        gpu_setup_logger.warning("   [CRITICAL GPU INIT FAIL][Patch AI Studio SmartFix] PyTorch or torch.cuda not available. Disabling GPU acceleration.")
        USE_GPU_ACCELERATION = False
        globals()['pynvml'] = None
        nvml_handle = None
        return

    try:
        if torch.cuda.is_available(): # type: ignore
            gpu_name = torch.cuda.get_device_name(0) if hasattr(torch.cuda, 'get_device_name') else "N/A" # type: ignore
            gpu_setup_logger.info(f"      พบ GPU (PyTorch): {gpu_name}. GPU Acceleration เปิดใช้งาน.")
            USE_GPU_ACCELERATION = True
            try:
                import pynvml as pynvml_local_module
                globals()['pynvml'] = pynvml_local_module
                gpu_setup_logger.info("[Patch - IMPORT ERROR FIX - Step 2 (GPU Setup)] Successfully imported and assigned pynvml module.")
                try:
                    pynvml.nvmlInit() # type: ignore
                    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # type: ignore
                    gpu_setup_logger.info("      เริ่มต้น pynvml สำหรับการตรวจสอบ GPU สำเร็จ.")
                except pynvml.NVMLError as e_nvml_init: # type: ignore # pragma: no cover
                    gpu_setup_logger.warning(f"      (Warning) NVML Initialization Error: {e_nvml_init}. GPU monitoring via pynvml disabled.")
                    nvml_handle = None
                    globals()['pynvml'] = None
                except Exception as e_nvml_generic: # pragma: no cover
                    gpu_setup_logger.warning(f"      (Warning) Generic NVML Error: {e_nvml_generic}. GPU monitoring via pynvml disabled.")
                    nvml_handle = None
                    globals()['pynvml'] = None
            except ImportError: # pragma: no cover
                gpu_setup_logger.warning("      (Warning) ไม่พบ pynvml library. GPU monitoring via pynvml disabled.")
                nvml_handle = None
                globals()['pynvml'] = None
            except Exception as e_pynvml_import: # pragma: no cover
                gpu_setup_logger.error(f"      (Error) An unexpected error occurred during pynvml import for GPU setup: {e_pynvml_import}")
                nvml_handle = None
                globals()['pynvml'] = None
        else: # pragma: no cover
            gpu_setup_logger.warning("      PyTorch ไม่พบ GPU หรือ CUDA ไม่พร้อมใช้งาน. GPU Acceleration ปิดใช้งาน.")
            USE_GPU_ACCELERATION = False
            globals()['pynvml'] = None
            nvml_handle = None
    except RuntimeError as e_triton: # pragma: no cover
        gpu_setup_logger.critical(f"   [CRITICAL GPU INIT FAIL][Patch AI Studio SmartFix] PyTorch C Extension/Triton initialization failed: {e_triton}. Disabling GPU acceleration.", exc_info=True)
        USE_GPU_ACCELERATION = False
        globals()['pynvml'] = None
        nvml_handle = None
    except Exception as e_gpu_setup: # pragma: no cover
        gpu_setup_logger.error(f"   (Error) เกิดข้อผิดพลาดระหว่างตรวจสอบ GPU: {e_gpu_setup}. GPU Acceleration ปิดใช้งาน.", exc_info=True)
        USE_GPU_ACCELERATION = False
        globals()['pynvml'] = None
        nvml_handle = None

# [Patch - IMPORT ERROR FIX - Step 1 & 4] Do NOT call setup_gpu_acceleration() here in global scope.
# It will be called from if __name__ == "__main__": block in Part 13.
logger.info("[Patch - IMPORT ERROR FIX - Step 4] setup_gpu_acceleration() call deferred to __main__ block.")


# --- System Status Printing Functions ---
def print_gpu_utilization(context_msg: str = "Current Status"): # pragma: no cover
    gpu_util_logger = logging.getLogger(f"{__name__}.print_gpu_utilization")
    ram_percent, ram_used_gb, ram_total_gb = "N/A", "N/A", "N/A"
    if psutil_imported and psutil:
        try:
            vmem = psutil.virtual_memory()
            ram_percent = f"{vmem.percent:.1f}%"
            ram_used_gb = f"{vmem.used / (1024**3):.1f}GB"
            ram_total_gb = f"{vmem.total / (1024**3):.1f}GB"
        except Exception as e_psutil_read:
            ram_percent = f"Error: {e_psutil_read}"
            gpu_util_logger.warning(f"   (Warning) Could not read RAM info from psutil: {e_psutil_read}")

    current_pynvml_module = globals().get('pynvml')
    current_nvml_handle = globals().get('nvml_handle')

    if USE_GPU_ACCELERATION and current_pynvml_module and current_nvml_handle:
        try:
            util = current_pynvml_module.nvmlDeviceGetUtilizationRates(current_nvml_handle)
            mem_info = current_pynvml_module.nvmlDeviceGetMemoryInfo(current_nvml_handle)
            gpu_util_logger.info(
                f"   [{context_msg}] GPU Util: {util.gpu}% | Mem: {util.memory}% "
                f"({mem_info.used // (1024**2)}MB / {mem_info.total // (1024**2)}MB) | "
                f"RAM: {ram_percent} ({ram_used_gb} / {ram_total_gb})"
            )
        except current_pynvml_module.NVMLError as e_nvml_read:
            gpu_util_logger.warning(f"   (Warning) Error reading GPU stats from pynvml: {e_nvml_read}. Disabling pynvml monitoring.")
            gpu_util_logger.info(
                f"   [{context_msg}] GPU Util: NVML Err | Mem: NVML Err: {str(e_nvml_read)[:50]}... | "
                f"RAM: {ram_percent} ({ram_used_gb} / {ram_total_gb})"
            )
            try:
                current_pynvml_module.nvmlShutdown()
            except Exception: pass
            globals()['pynvml'] = None
            globals()['nvml_handle'] = None
        except Exception as e_generic_gpu_read:
            gpu_util_logger.warning(f"   (Warning) Generic error reading GPU stats: {e_generic_gpu_read}.")
            gpu_util_logger.info(
                f"   [{context_msg}] GPU Util: Error | Mem: Error | "
                f"RAM: {ram_percent} ({ram_used_gb} / {ram_total_gb})"
            )
    elif USE_GPU_ACCELERATION and not current_pynvml_module:
        gpu_util_logger.info(
            f"   [{context_msg}] GPU Util: pynvml N/A | Mem: pynvml N/A | "
            f"RAM: {ram_percent} ({ram_used_gb} / {ram_total_gb})"
        )
    else:
        gpu_util_logger.info(
            f"   [{context_msg}] GPU Util: Disabled | Mem: Disabled | "
            f"RAM: {ram_percent} ({ram_used_gb} / {ram_total_gb})"
        )

def show_system_status(context_msg: str = "System Status"): # pragma: no cover
    sys_status_logger = logging.getLogger(f"{__name__}.show_system_status")
    ram_usage_str = "RAM: N/A"
    if psutil_imported and psutil:
        try:
            vmem = psutil.virtual_memory()
            ram_usage_str = f"RAM: {vmem.percent:.1f}% ({vmem.used / (1024**3):.1f}GB / {vmem.total / (1024**3):.1f}GB)"
        except Exception as e_psutil_sys:
            ram_usage_str = f"RAM Error: {e_psutil_sys}"
            sys_status_logger.warning(f"   (Warning) Could not get RAM info for system status: {e_psutil_sys}")

    gpu_info_str_list = []
    if gputil_imported and GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu_item in gpus:
                    gpu_info_str_list.append(
                        f"GPU {gpu_item.id} {gpu_item.name} | Load: {gpu_item.load*100:.1f}% | "
                        f"Mem: {gpu_item.memoryUtil*100:.1f}% ({gpu_item.memoryUsed:.0f}MB/{gpu_item.memoryTotal:.0f}MB)"
                    )
            else:
                gpu_info_str_list.append("No GPU found by GPUtil")
        except Exception as e_gputil:
            gpu_info_str_list.append(f"GPUtil Error: {e_gputil}")
            sys_status_logger.warning(f"   (Warning) Error getting GPU info from GPUtil: {e_gputil}")
    else:
        gpu_info_str_list.append("GPUtil N/A")

    final_status_msg = f"   [{context_msg}] {ram_usage_str} | {' | '.join(gpu_info_str_list)}"
    sys_status_logger.info(final_status_msg)

# --- Minimal Test Function (as requested in prompt) ---
def minimal_test_function(): # pragma: no cover
    return "Minimal test function from gold_ai_module executed successfully!"

logger.info(f"Part 1/15 (Setup, Imports, Globals) Loaded. Script Version: {MINIMAL_SCRIPT_VERSION}")
logger.info("[Patch - IMPORT ERROR FIX - Step 5 (Review)] Developer to review try-except ImportError blocks for specificity.")
# === END OF PART 1/15 ===
# ==============================================================================
# === START OF PART 2/15 ===
# ==============================================================================
# === PART 2: Core Parameters & Strategy Settings (DEPRECATED - Enterprise Refactor) ===
# ==============================================================================
# <<< MODIFIED: Enterprise Refactor - This entire part is now effectively empty. >>>
# <<< All parameters previously defined here will be loaded from config.yaml >>>
# <<< into an instance of StrategyConfig within the main() function. >>>
# <<< Some constants related to feature engineering might be defined directly in Part 6 (formerly Part 5). >>>

import logging # Ensure logging is available

# Module-level logger for this part, though it will be minimal.
part2_logger = logging.getLogger(f"{__name__}.Part2_DeprecatedParams")

part2_logger.info("Part 2: Core Parameters & Strategy Settings (DEPRECATED - Parameters now in StrategyConfig from YAML).")

# --- Feature Engineering Constants ---
# These were previously in Part 2. They are more relevant to feature engineering (Part 6, formerly Part 5)
# and will be defined there if they are true constants not suitable for StrategyConfig.
# Examples of what was here:
# TIMEFRAME_MINUTES_M15 = 15
# ROLLING_Z_WINDOW_M1 = 300
# ATR_ROLLING_AVG_PERIOD = 50
# PATTERN_BREAKOUT_Z_THRESH = 2.0
# M15_TREND_EMA_FAST = 50
# SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)} # This will be in StrategyConfig

# --- Other Constants ---
# MAX_NAT_RATIO_THRESHOLD = 0.05 # Data quality threshold, might not be in strategy config.
#                                # If it's a fixed system constraint, it can remain a global constant,
#                                # possibly defined in Part 1 or near its usage. For now, assume it's handled.

# --- Fold-Specific Configuration ---
# ENTRY_CONFIG_PER_FOLD = { ... }
# This structure will be re-evaluated. If still needed for fold-specific overrides
# not covered by the main StrategyConfig, it might be loaded differently or
# passed directly to relevant functions. It's removed from global scope here.
# The `fold_config` argument in `run_backtest_simulation_v34` can handle this.

part2_logger.warning("Part 2: Is DEPRECATED. All tunable strategy parameters should be managed by StrategyConfig.")
part2_logger.info("Any true constants (e.g., for feature engineering) will be defined in their relevant sections or Part 1.")
# === END OF PART 2/15 ===
# === START OF PART 3/15 ===
# ==============================================================================
# === PART 3: Enterprise Refactor Classes & Config Loader (v4.9.15 - StrategyConfig with Default Signal Thresholds) ===
# ==============================================================================
# <<< MODIFIED: Added default signal calculation thresholds to StrategyConfig. >>>
# <<< MODIFIED: [Patch] Added default attributes for dynamic_tp2_multiplier and adjust_lot_tp2_boost to StrategyConfig. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.1] Ensured all default attributes for dynamic TP2 and lot boost are present. >>>

import logging  # Already imported in Part 1, but good practice for clarity if part is moved
import yaml  # Already imported in Part 1
import os  # Already imported in Part 1
import pandas as pd  # Already imported in Part 1
import numpy as np  # Already imported in Part 1
# datetime is imported in Part 1 as `import datetime`
from typing import Union, Optional, Callable, Any, Dict, List, Tuple # Added for type hints


# --- Strategy Configuration Class ---
class StrategyConfig:
    """
    Stores all tunable strategy parameters, loaded from a configuration file
    or using default values.
    """

    def __init__(self, config_dict: Dict[str, Any]): # type: ignore
        self.logger = logging.getLogger(f"{__name__}.StrategyConfig")
        self.logger.debug(f"Initializing StrategyConfig with dict: {config_dict}")

        # --- Risk & Lot Management ---
        self.risk_per_trade: float = config_dict.get("risk_per_trade", 0.01)
        self.max_lot: float = config_dict.get("max_lot", 5.0)
        self.min_lot: float = config_dict.get("min_lot", 0.01)

        # --- Kill Switch & Recovery ---
        self.kill_switch_dd: float = config_dict.get("kill_switch_dd", 0.20)
        self.soft_kill_dd: float = config_dict.get("soft_kill_dd", 0.15)
        self.kill_switch_consecutive_losses: int = config_dict.get("kill_switch_consecutive_losses", 7)
        self.recovery_mode_consecutive_losses: int = config_dict.get("recovery_mode_consecutive_losses", 4)
        self.recovery_mode_lot_multiplier: float = config_dict.get("recovery_mode_lot_multiplier", 0.5)

        # --- Trade Holding & Timing ---
        self.max_holding_bars: Optional[int] = config_dict.get("max_holding_bars", 24)
        if "max_holding_bars" in config_dict and config_dict["max_holding_bars"] is None:
            self.max_holding_bars = None

        # --- Forced Entry System ---
        self.enable_forced_entry: bool = config_dict.get("enable_forced_entry", True)
        self.forced_entry_cooldown_minutes: int = config_dict.get("forced_entry_cooldown_minutes", 240)
        self.forced_entry_score_min: float = config_dict.get("forced_entry_score_min", 1.0)
        self.forced_entry_max_atr_mult: float = config_dict.get("forced_entry_max_atr_mult", 2.5)
        self.forced_entry_min_gain_z_abs: float = config_dict.get("forced_entry_min_gain_z_abs", 1.0)
        self.forced_entry_allowed_regimes: List[str] = config_dict.get("forced_entry_allowed_regimes", ["Normal", "Breakout", "StrongTrend"])
        self.fe_ml_filter_threshold: float = config_dict.get("fe_ml_filter_threshold", 0.40)
        self.forced_entry_max_consecutive_losses: int = config_dict.get("forced_entry_max_consecutive_losses", 2)

        # --- Partial Take Profit ---
        self.enable_partial_tp: bool = config_dict.get("enable_partial_tp", True)
        self.partial_tp_levels: List[Dict[str, float]] = config_dict.get("partial_tp_levels", [{"r_multiple": 0.8, "close_pct": 0.5}])
        self.partial_tp_move_sl_to_entry: bool = config_dict.get("partial_tp_move_sl_to_entry", True)

        # --- Re-Entry System ---
        self.use_reentry: bool = config_dict.get("use_reentry", True)
        self.reentry_cooldown_bars: int = config_dict.get("reentry_cooldown_bars", 1)
        self.reentry_min_proba_thresh: float = config_dict.get("reentry_min_proba_thresh", 0.55)
        self.reentry_cooldown_after_tp_minutes: int = config_dict.get("reentry_cooldown_after_tp_minutes", 30)


        # --- Spike Guard ---
        self.enable_spike_guard: bool = config_dict.get("enable_spike_guard", True)
        self.spike_guard_score_threshold: float = config_dict.get("spike_guard_score_threshold", 0.75)
        self.spike_guard_london_patterns: List[str] = config_dict.get("spike_guard_london_patterns", ["Breakout", "StrongTrend"])

        # --- Machine Learning Thresholds & Features ---
        self.meta_min_proba_thresh: float = config_dict.get("meta_min_proba_thresh", 0.55)
        self.meta_classifier_features: List[str] = config_dict.get("meta_classifier_features", [])
        self.spike_model_features: List[str] = config_dict.get("spike_model_features", [])
        self.cluster_model_features: List[str] = config_dict.get("cluster_model_features", [])
        self.shap_importance_threshold: float = config_dict.get("shap_importance_threshold", 0.01)
        self.shap_noise_threshold: float = config_dict.get("shap_noise_threshold", 0.005)

        # --- Backtesting General Parameters ---
        self.initial_capital: float = config_dict.get("initial_capital", 100.0)
        self.commission_per_001_lot: float = config_dict.get("commission_per_001_lot", 0.10)
        self.spread_points: float = config_dict.get("spread_points", 2.0)
        self.point_value: float = config_dict.get("point_value", 0.1)
        self.ib_commission_per_lot: float = config_dict.get("ib_commission_per_lot", 7.0)

        # --- Paths & File Names ---
        self.n_walk_forward_splits: int = config_dict.get("n_walk_forward_splits", 5)
        self.output_base_dir: str = config_dict.get("output_base_dir", "/content/drive/MyDrive/new_enterprise_output")
        self.output_dir_name: str = config_dict.get("output_dir_name", "gold_ai_run")
        self.data_file_path_m15: str = config_dict.get("data_file_path_m15", "/content/drive/MyDrive/new/XAUUSD_M15.csv")
        self.data_file_path_m1: str = config_dict.get("data_file_path_m1", "/content/drive/MyDrive/new/XAUUSD_M1.csv")
        self.config_file_path: str = config_dict.get("config_file_path", "config.yaml")
        self.meta_classifier_filename: str = config_dict.get("meta_classifier_filename", "meta_classifier.pkl")
        self.spike_model_filename: str = config_dict.get("spike_model_filename", "meta_classifier_spike.pkl")
        self.cluster_model_filename: str = config_dict.get("cluster_model_filename", "meta_classifier_cluster.pkl")
        self.base_train_trade_log_path: str = config_dict.get("base_train_trade_log_path", os.path.join(self.output_base_dir, self.output_dir_name, "trade_log_v32_walkforward"))
        self.base_train_m1_data_path: str = config_dict.get("base_train_m1_data_path", os.path.join(self.output_base_dir, self.output_dir_name, "final_data_m1_v32_walkforward"))
        self.trade_log_filename_prefix: str = config_dict.get("trade_log_filename_prefix", "trade_log")
        self.summary_filename_prefix: str = config_dict.get("summary_filename_prefix", "run_summary")

        # --- Adaptive TSL parameters ---
        self.adaptive_tsl_start_atr_mult: float = config_dict.get("adaptive_tsl_start_atr_mult", 1.5)
        self.adaptive_tsl_default_step_r: float = config_dict.get("adaptive_tsl_default_step_r", 0.5)
        self.adaptive_tsl_high_vol_ratio: float = config_dict.get("adaptive_tsl_high_vol_ratio", 1.8)
        self.adaptive_tsl_high_vol_step_r: float = config_dict.get("adaptive_tsl_high_vol_step_r", 1.0)
        self.adaptive_tsl_low_vol_ratio: float = config_dict.get("adaptive_tsl_low_vol_ratio", 0.75)
        self.adaptive_tsl_low_vol_step_r: float = config_dict.get("adaptive_tsl_low_vol_step_r", 0.3)

        # --- Base TP/BE parameters ---
        self.base_tp_multiplier: float = config_dict.get("base_tp_multiplier", 1.8)
        self.base_be_sl_r_threshold: float = config_dict.get("base_be_sl_r_threshold", 1.0)
        self.default_sl_multiplier: float = config_dict.get("default_sl_multiplier", 1.5)

        # --- Min signal score ---
        self.min_signal_score_entry: float = config_dict.get("min_signal_score_entry", 2.0)

        # --- Session Times (UTC) ---
        self.session_times_utc: Dict[str, Tuple[int, int]] = config_dict.get("session_times_utc", {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)})

        # --- Feature Engineering Constants ---
        self.timeframe_minutes_m15: int = config_dict.get("timeframe_minutes_m15", 15)
        self.timeframe_minutes_m1: int = config_dict.get("timeframe_minutes_m1", 1)
        self.rolling_z_window_m1: int = config_dict.get("rolling_z_window_m1", 300)
        self.atr_rolling_avg_period: int = config_dict.get("atr_rolling_avg_period", 50)
        self.pattern_breakout_z_thresh: float = config_dict.get("pattern_breakout_z_thresh", 2.0)
        self.pattern_reversal_body_ratio: float = config_dict.get("pattern_reversal_body_ratio", 0.5)
        self.pattern_strong_trend_z_thresh: float = config_dict.get("pattern_strong_trend_z_thresh", 1.0)
        self.pattern_choppy_candle_ratio: float = config_dict.get("pattern_choppy_candle_ratio", 0.3)
        self.pattern_choppy_wick_ratio: float = config_dict.get("pattern_choppy_wick_ratio", 0.6)
        self.m15_trend_ema_fast: int = config_dict.get("m15_trend_ema_fast", 50)
        self.m15_trend_ema_slow: int = config_dict.get("m15_trend_ema_slow", 200)
        self.m15_trend_rsi_period: int = config_dict.get("m15_trend_rsi_period", 14)
        self.m15_trend_rsi_up: int = config_dict.get("m15_trend_rsi_up", 52)
        self.m15_trend_rsi_down: int = config_dict.get("m15_trend_rsi_down", 48)
        self.m15_trend_merge_tolerance_minutes: int = config_dict.get("m15_trend_merge_tolerance_minutes", 30)

        # --- Default Signal Calculation Thresholds (NEWLY ADDED for Part 6) ---
        self.default_gain_z_thresh_fold: float = config_dict.get("default_gain_z_thresh_fold", 0.3)
        self.default_rsi_thresh_buy_fold: int = config_dict.get("default_rsi_thresh_buy_fold", 50)
        self.default_rsi_thresh_sell_fold: int = config_dict.get("default_rsi_thresh_sell_fold", 50)
        self.default_volatility_max_fold: float = config_dict.get("default_volatility_max_fold", 4.0)
        self.default_ignore_rsi_scoring_fold: bool = config_dict.get("default_ignore_rsi_scoring_fold", False)


        # --- Model Training Parameters ---
        self.enable_dynamic_feature_selection: bool = config_dict.get("enable_dynamic_feature_selection", True)
        self.feature_selection_method: str = config_dict.get("feature_selection_method", 'shap')
        self.prelim_model_params: Optional[Dict[str, Any]] = config_dict.get("prelim_model_params", None)
        self.enable_optuna_tuning: bool = config_dict.get("enable_optuna_tuning", False)
        self.optuna_n_trials: int = config_dict.get("optuna_n_trials", 50)
        self.optuna_cv_splits: int = config_dict.get("optuna_cv_splits", 5)
        self.optuna_metric: str = config_dict.get("optuna_metric", "AUC")
        self.optuna_direction: str = config_dict.get("optuna_direction", "maximize")
        self.catboost_gpu_ram_part: float = config_dict.get("catboost_gpu_ram_part", 0.95)
        self.optuna_n_jobs: int = config_dict.get("optuna_n_jobs", -1)
        self.sample_size_train: int = config_dict.get("sample_size_train", 60000)
        self.features_to_drop_train: Optional[List[str]] = config_dict.get("features_to_drop_train", None)
        self.early_stopping_rounds: int = config_dict.get("early_stopping_rounds", 200)
        self.permutation_importance_threshold: float = config_dict.get("permutation_importance_threshold", 0.001)
        self.catboost_iterations: int = config_dict.get("catboost_iterations", 3000)
        self.catboost_learning_rate: float = config_dict.get("catboost_learning_rate", 0.01)
        self.catboost_depth: int = config_dict.get("catboost_depth", 4)
        self.catboost_l2_leaf_reg: int = config_dict.get("catboost_l2_leaf_reg", 30)
        self.lag_features_config: Optional[Dict[str, Any]] = config_dict.get("lag_features_config", None)

        # --- Auto-Train Specific Parameters ---
        self.auto_train_enable_optuna: bool = config_dict.get("auto_train_enable_optuna", False)
        self.auto_train_enable_dynamic_features: bool = config_dict.get("auto_train_enable_dynamic_features", True)
        self.auto_train_spike_filter_threshold: float = config_dict.get("auto_train_spike_filter_threshold", 0.6)
        self.auto_train_cluster_filter_value: int = config_dict.get("auto_train_cluster_filter_value", 2)

        # --- Drift Detection Parameters ---
        self.drift_wasserstein_threshold: float = config_dict.get("drift_wasserstein_threshold", 0.1)
        self.drift_ttest_alpha: float = config_dict.get("drift_ttest_alpha", 0.05)
        self.drift_min_data_points: int = config_dict.get("drift_min_data_points", 10)
        self.drift_alert_features: List[str] = config_dict.get("drift_alert_features", ['Gain_Z', 'ATR_14', 'Candle_Speed', 'RSI'])
        self.drift_warning_factor: float = config_dict.get("drift_warning_factor", 1.5)
        self.drift_adjustment_sensitivity: float = config_dict.get("drift_adjustment_sensitivity", 1.0)
        self.drift_max_gain_z_thresh: float = config_dict.get("drift_max_gain_z_thresh", 3.0)
        self.drift_min_gain_z_thresh: float = config_dict.get("drift_min_gain_z_thresh", 0.1)
        self.m1_features_for_drift: Optional[List[str]] = config_dict.get("m1_features_for_drift", None)

        # --- Multi-Fund & WFV Parameters ---
        self.multi_fund_mode: bool = config_dict.get("multi_fund_mode", False)
        self.fund_profiles: Dict[str, Any] = config_dict.get("fund_profiles", {})
        self.default_fund_name: str = config_dict.get("default_fund_name", "DEFAULT_FUND")
        self.default_fund_name_for_prep_fallback: str = config_dict.get("default_fund_name_for_prep_fallback", "PREP_DEFAULT")
        self.entry_config_per_fold: Dict[int, Any] = config_dict.get("entry_config_per_fold", {})
        self.current_fund_name_for_logging: str = self.default_fund_name

        # --- Other System/Run Control Parameters ---
        self.use_gpu_acceleration: bool = config_dict.get("use_gpu_acceleration", True)
        self.train_meta_model_before_run: bool = config_dict.get("train_meta_model_before_run", True)
        self.max_nat_ratio_threshold: float = config_dict.get("max_nat_ratio_threshold", 0.05)
        self.min_slippage_points: float = config_dict.get("min_slippage_points", -5.0)
        self.max_slippage_points: float = config_dict.get("max_slippage_points", -1.0)
        self.ttp2_atr_threshold_activate: float = config_dict.get("ttp2_atr_threshold_activate", 4.0)
        self.soft_cooldown_lookback: int = config_dict.get("soft_cooldown_lookback", 10)

        # <<< MODIFIED: [Patch] Added default attributes for dynamic_tp2_multiplier and adjust_lot_tp2_boost. >>>
        # <<< MODIFIED: [Patch AI Studio v4.9.1] Ensured all default attributes for dynamic TP2 and lot boost are present. >>>
        self.tp2_dynamic_vol_high_ratio: float = config_dict.get("tp2_dynamic_vol_high_ratio", self.adaptive_tsl_high_vol_ratio)
        self.tp2_dynamic_vol_low_ratio: float = config_dict.get("tp2_dynamic_vol_low_ratio", self.adaptive_tsl_low_vol_ratio)
        self.tp2_dynamic_high_vol_boost: float = config_dict.get("tp2_dynamic_high_vol_boost", 1.2)
        self.tp2_dynamic_low_vol_reduce: float = config_dict.get("tp2_dynamic_low_vol_reduce", 0.8)
        self.tp2_dynamic_min_multiplier: float = config_dict.get("tp2_dynamic_min_multiplier", self.base_tp_multiplier * 0.5)
        self.tp2_dynamic_max_multiplier: float = config_dict.get("tp2_dynamic_max_multiplier", self.base_tp_multiplier * 2.0)
        self.tp2_boost_lookback_trades: int = config_dict.get("tp2_boost_lookback_trades", 3)
        self.tp2_boost_tp_count_threshold: int = config_dict.get("tp2_boost_tp_count_threshold", 2)
        self.tp2_boost_multiplier: float = config_dict.get("tp2_boost_multiplier", 1.10)
        # <<< END OF MODIFIED [Patch AI Studio v4.9.1] >>>

        self.logger.info(f"StrategyConfig initialized. Risk/Trade: {self.risk_per_trade:.3f}, Max Hold: {self.max_holding_bars} bars, Kill DD: {self.kill_switch_dd:.2%}")
        self.logger.debug(f"  Feature Eng Params: M15 EMA Fast={self.m15_trend_ema_fast}, M1 Rolling Z Window={self.rolling_z_window_m1}")
        self.logger.debug(f"  Default Signal Thresh: GainZ={self.default_gain_z_thresh_fold}, RSI Buy={self.default_rsi_thresh_buy_fold}")
        self.logger.debug(f"  Dynamic TP2 Params: HighVolBoost={self.tp2_dynamic_high_vol_boost}, LotBoostMult={self.tp2_boost_multiplier}")


# --- Risk Management Class ---
class RiskManager:
    """
    Manages risk aspects like drawdown, kill switches, and soft kill switches.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.dd_peak: Optional[float] = None
        self.soft_kill_active: bool = False
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        self.logger.info(f"RiskManager initialized. Hard Kill DD: {self.config.kill_switch_dd:.2%}, Soft Kill DD: {self.config.soft_kill_dd:.2%}")

    def update_drawdown(self, current_equity: float) -> float:
        if self.dd_peak is None or pd.isna(self.dd_peak):
            self.dd_peak = current_equity
            self.logger.debug(f"Drawdown peak initialized to current equity: {current_equity:.2f}")

        if pd.isna(current_equity):
            self.logger.warning("Current equity is NaN in update_drawdown. Cannot calculate drawdown.")
            return 0.0

        self.dd_peak = max(self.dd_peak, current_equity)

        if self.dd_peak <= 1e-9: # type: ignore
            drawdown = 0.0
            self.logger.debug(f"Drawdown peak is near zero ({self.dd_peak:.2f}). Drawdown set to 0.") # type: ignore
        else:
            drawdown = 1.0 - (current_equity / self.dd_peak) # type: ignore

        self.logger.debug(f"Drawdown updated. Equity={current_equity:.2f}, Peak={self.dd_peak:.2f}, DD={drawdown:.4f}") # type: ignore

        if drawdown >= self.config.soft_kill_dd:
            if not self.soft_kill_active:
                self.logger.info(f"[RISK] Soft Kill Switch ACTIVATED. DD={drawdown:.4f} >= Threshold={self.config.soft_kill_dd:.4f}")
            self.soft_kill_active = True
        else:
            if self.soft_kill_active:
                self.logger.info(f"[RISK] Soft Kill Switch DEACTIVATED. DD={drawdown:.4f} < Threshold={self.config.soft_kill_dd:.4f}")
            self.soft_kill_active = False

        if drawdown >= self.config.kill_switch_dd:
            self.logger.critical(f"[RISK - KILL SWITCH] Max Drawdown Threshold Hit. Equity={current_equity:.2f}, Peak={self.dd_peak:.2f}, DD={drawdown:.4f}, Threshold={self.config.kill_switch_dd:.4f}") # type: ignore
            raise RuntimeError(f"[KILL SWITCH] Max Drawdown Threshold Hit ({drawdown:.2%}). System Stopped.")
        return drawdown

    def check_consecutive_loss_kill(self, consecutive_losses: int) -> bool:
        if consecutive_losses >= self.config.kill_switch_consecutive_losses:
            self.logger.critical(f"[RISK - KILL SWITCH] Consecutive Losses Threshold Hit. Losses={consecutive_losses} >= Threshold={self.config.kill_switch_consecutive_losses}")
            return True
        return False

    def is_trading_allowed(self) -> bool:
        if self.soft_kill_active:
            self.logger.debug("[RISK] Trading blocked: Soft Kill Switch is active.")
            return False
        return True


# --- Trade Management Class ---
class TradeManager:
    """
    Manages trade-related logic, including Forced Entry.
    """

    def __init__(self, config: StrategyConfig, risk_manager: RiskManager):
        self.config = config
        self.risk_manager = risk_manager
        self.last_trade_time: Optional[pd.Timestamp] = None
        self.consecutive_forced_losses: int = 0
        self.logger = logging.getLogger(f"{__name__}.TradeManager")
        self.logger.info(f"TradeManager initialized. FE Cooldown: {self.config.forced_entry_cooldown_minutes} min, FE Score: {self.config.forced_entry_score_min}, FE Max Losses: {self.config.forced_entry_max_consecutive_losses}")

    def update_last_trade_time(self, trade_time: pd.Timestamp):
        if pd.isna(trade_time):
            self.logger.warning("Attempted to update last_trade_time with NaT.")
            return
        self.last_trade_time = trade_time
        self.logger.debug(f"Last trade time updated to: {self.last_trade_time}")

    def update_forced_entry_result(self, is_loss: bool):
        if is_loss:
            self.consecutive_forced_losses += 1
            self.logger.info(f"[TRADE_MGR] Forced entry resulted in a loss. Consecutive forced losses: {self.consecutive_forced_losses}")
        else:
            if self.consecutive_forced_losses > 0:
                self.logger.info(f"[TRADE_MGR] Forced entry was not a loss. Resetting consecutive forced losses from {self.consecutive_forced_losses} to 0.")
            self.consecutive_forced_losses = 0

    def should_force_entry(self, current_time: pd.Timestamp, signal_score: Optional[float],
                           current_atr: Optional[float], avg_atr: Optional[float],
                           gain_z: Optional[float], pattern_label: Optional[str]) -> bool:
        if not self.config.enable_forced_entry:
            return False

        if not self.risk_manager.is_trading_allowed():
            self.logger.debug("[TRADE_MGR] Forced entry check: Trading currently blocked by RiskManager (Soft Kill).")
            return False

        time_since_last_trade_minutes = float('inf')
        if self.last_trade_time is not None and not pd.isna(self.last_trade_time):
            if pd.isna(current_time):
                self.logger.warning("[TRADE_MGR] Forced entry check: current_time is NaT. Cannot calculate time since last trade.")
                return False
            try:
                time_since_last_trade_minutes = (current_time - self.last_trade_time).total_seconds() / 60.0
                if time_since_last_trade_minutes < self.config.forced_entry_cooldown_minutes:
                    self.logger.debug(f"[TRADE_MGR] Forced entry check: Cooldown active. {time_since_last_trade_minutes:.1f} min < {self.config.forced_entry_cooldown_minutes} min.")
                    return False
            except TypeError as te: # pragma: no cover
                self.logger.error(f"[TRADE_MGR] Forced entry check: TypeError calculating time_since_last_trade (current_time: {current_time}, last_trade_time: {self.last_trade_time}): {te}")
                return False
        else:
            self.logger.debug("[TRADE_MGR] Forced entry check: No last trade time recorded, cooldown condition met.")


        if pd.isna(signal_score) or abs(signal_score if signal_score is not None else 0.0) < self.config.forced_entry_score_min:
            self.logger.debug(f"[TRADE_MGR] Forced entry check: Signal score ({signal_score}) below threshold ({self.config.forced_entry_score_min}).")
            return False

        if self.consecutive_forced_losses >= self.config.forced_entry_max_consecutive_losses:
            self.logger.info(f"[TRADE_MGR] Forced entry blocked: Max consecutive forced losses ({self.consecutive_forced_losses}) reached threshold ({self.config.forced_entry_max_consecutive_losses}).")
            return False

        if pd.isna(current_atr) or pd.isna(avg_atr) or pd.isna(gain_z) or pd.isna(pattern_label):
            self.logger.debug("[TRADE_MGR] Forced entry check: Market condition data (ATR, GainZ, Pattern) has NaNs. Skipping these specific market condition checks.")
        else:
            if avg_atr is not None and avg_atr > 1e-9 and current_atr is not None and (current_atr / avg_atr) > self.config.forced_entry_max_atr_mult:
                self.logger.debug(f"[TRADE_MGR] Forced entry check: ATR ratio ({current_atr/avg_atr:.2f if avg_atr is not None and avg_atr > 1e-9 else 'N/A'}) above max ({self.config.forced_entry_max_atr_mult}).")
                return False
            if gain_z is not None and abs(gain_z) < self.config.forced_entry_min_gain_z_abs:
                self.logger.debug(f"[TRADE_MGR] Forced entry check: Abs Gain_Z ({abs(gain_z):.2f}) below min ({self.config.forced_entry_min_gain_z_abs}).")
                return False
            if pattern_label not in self.config.forced_entry_allowed_regimes:
                self.logger.debug(f"[TRADE_MGR] Forced entry check: Pattern Label '{pattern_label}' not in allowed regimes {self.config.forced_entry_allowed_regimes}.")
                return False

        self.logger.info(f"[TRADE_MGR] Conditions MET for Forced Entry. Time since last: {time_since_last_trade_minutes:.1f} min, Score: {signal_score:.2f if signal_score is not None else 'N/A'}")
        return True


# --- Configuration Loading Function ---
def load_config_from_yaml(path: str = "config.yaml") -> StrategyConfig:
    config_loader_logger = logging.getLogger(f"{__name__}.load_config_from_yaml")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        if raw_config is None:
            config_loader_logger.warning(f"Config file '{path}' is empty or invalid. Using default config values.")
            return StrategyConfig({})
        config_loader_logger.info(f"Successfully loaded configuration from: {path}")
        return StrategyConfig(raw_config)
    except FileNotFoundError:
        config_loader_logger.warning(f"[Warning] Config file '{path}' not found. Using default config values.")
        return StrategyConfig({})
    except yaml.YAMLError as e_yaml: # pragma: no cover
        config_loader_logger.error(f"[Error] Failed to parse YAML from '{path}': {e_yaml}. Using default config values.")
        return StrategyConfig({})
    except Exception as e: # pragma: no cover
        config_loader_logger.error(f"[Error] Failed to load config from '{path}', using defaults. Reason: {e}", exc_info=True)
        return StrategyConfig({})


# --- Holding Period Exit Function ---
def should_exit_due_to_holding(current_bar_idx: int, entry_bar_idx: int, max_holding_bars_config: Optional[int]) -> bool:
    if max_holding_bars_config is None or not isinstance(max_holding_bars_config, int) or max_holding_bars_config <= 0:
        return False
    return (current_bar_idx - entry_bar_idx) >= max_holding_bars_config


logger.info("Part 3 (New): Enterprise Classes & Config Loader Implemented with Feature Engineering and Default Signal Thresholds.")
# === END OF PART 3/15 ===
# === START OF PART 4/15 ===
# ==============================================================================
# === PART 4: Helper Functions (Setup, Utils, Font, Config) (v4.9.0 - Enterprise Refactor) ===
# ==============================================================================
# <<< MODIFIED: Enterprise Refactor - load_app_config is now DEPRECATED. >>>
# <<< safe_get_global is marked as DEPRECATED. Loggers are made more specific. >>>

import logging # Already imported
import os # Already imported
import sys # Already imported
import subprocess # Already imported
import traceback # Already imported
import pandas as pd # Already imported
import numpy as np # Already imported
import json # Already imported
import gzip # Already imported
import matplotlib.pyplot as plt # Already imported
import matplotlib.font_manager as fm # Already imported
from IPython import get_ipython # type: ignore # Already imported
import requests # Already imported
import datetime # Standard datetime import # Already imported

# --- Helper for Safe Global Access (DEPRECATED) ---
def safe_get_global(var_name, default_value):
    """
    DEPRECATED: Safely retrieves a global variable from the current module's scope.
    Use StrategyConfig for managing configurations. Kept for compatibility during transition.
    """
    # Module-level logger is fine here as this function is general purpose.
    # logger.debug(f"[safe_get_global - DEPRECATED] Attempting to get global '{var_name}'")
    try:
        return globals().get(var_name, default_value)
    except Exception as e:
        logger.error(f"   (Error) Unexpected error in safe_get_global for '{var_name}': {e}", exc_info=True)
        return default_value

# --- Directory Setup Helper ---
def setup_output_directory(base_dir: str, dir_name: str) -> str:
    """
    Creates the output directory if it doesn't exist and checks write permissions.
    """
    output_path = os.path.join(base_dir, dir_name)
    # Using module-level logger here as this function might be called early.
    setup_dir_logger = logging.getLogger(f"{__name__}.setup_output_directory")
    setup_dir_logger.info(f"   (Setup) กำลังตรวจสอบ/สร้าง Output Directory: {output_path}")
    try:
        os.makedirs(output_path, exist_ok=True)
        setup_dir_logger.info(f"      -> Directory exists or was created.")
        test_file_path = os.path.join(output_path, ".write_test")
        with open(test_file_path, "w", encoding='utf-8') as f:
            f.write("test")
        os.remove(test_file_path)
        setup_dir_logger.info(f"      -> การเขียนไฟล์ทดสอบสำเร็จ.")
        return output_path
    except OSError as e:
        setup_dir_logger.error(f"   (Error) ไม่สามารถสร้างหรือเขียนใน Output Directory '{output_path}': {e}", exc_info=True)
        sys.exit(f"   ออก: ปัญหาการเข้าถึง Output Directory ({output_path}).")
    except Exception as e:
        setup_dir_logger.error(f"   (Error) เกิดข้อผิดพลาดที่ไม่คาดคิดระหว่างตั้งค่า Output Directory '{output_path}': {e}", exc_info=True)
        sys.exit(f"   ออก: ข้อผิดพลาดร้ายแรงในการตั้งค่า Output Directory ({output_path}).")

# --- Font Setup Helpers ---
def set_thai_font(font_name: str = "Loma") -> bool:
    """
    Attempts to set the specified Thai font for Matplotlib using findfont.
    Prioritizes specified font, then searches common fallbacks.
    """
    target_font_path = None
    actual_font_name = None
    preferred_fonts = [font_name] + ["TH Sarabun New", "THSarabunNew", "Garuda", "Norasi", "Kinnari", "Waree", "Laksaman", "Loma"]
    preferred_fonts = list(dict.fromkeys(preferred_fonts)) # Remove duplicates, keep order
    font_logger = logging.getLogger(f"{__name__}.set_thai_font")
    font_logger.info(f"   [Font Check] Searching for preferred fonts: {preferred_fonts}")

    for pref_font in preferred_fonts:
        try:
            found_path = fm.findfont(pref_font, fallback_to_default=False) # type: ignore
            if found_path and os.path.exists(found_path):
                target_font_path = found_path
                prop = fm.FontProperties(fname=target_font_path) # type: ignore
                actual_font_name = prop.get_name()
                font_logger.info(f"      -> Found font: '{actual_font_name}' (requested: '{pref_font}') at path: {target_font_path}")
                break
        except ValueError:
            font_logger.debug(f"      -> Font '{pref_font}' not found by findfont.")
        except Exception as e_find:
            font_logger.warning(f"      -> Error finding font '{pref_font}': {e_find}")

    if target_font_path and actual_font_name:
        try:
            plt.rcParams['font.family'] = actual_font_name # type: ignore
            plt.rcParams['axes.unicode_minus'] = False
            font_logger.info(f"   Attempting to set default font to '{actual_font_name}'.")
            fig_test, ax_test = plt.subplots(figsize=(0.5, 0.5))
            ax_test.set_title(f"ทดสอบ ({actual_font_name})", fontname=actual_font_name)
            plt.close(fig_test)
            font_logger.info(f"      -> Font '{actual_font_name}' set and tested successfully.")
            return True
        except Exception as e_set:
            font_logger.warning(f"      -> (Warning) Font '{actual_font_name}' set, but test failed: {e_set}")
            try:
                plt.rcParams['font.family'] = 'DejaVu Sans' # type: ignore
                font_logger.info("         -> Reverted to 'DejaVu Sans' due to test failure.")
            except Exception as e_revert:
                font_logger.warning(f"         -> Failed to revert font to DejaVu Sans: {e_revert}")
            return False
    else:
        font_logger.warning(f"   (Warning) Could not find any suitable Thai fonts ({preferred_fonts}) using findfont.")
        return False

def setup_fonts(output_dir: str | None = None): # output_dir is not used currently
    """
    Sets up Thai fonts for Matplotlib plots.
    Attempts to find preferred fonts, installs 'fonts-thai-tlwg' on Colab if needed.
    """
    font_setup_logger = logging.getLogger(f"{__name__}.setup_fonts")
    font_setup_logger.info("\n(Processing) Setting up Thai font for plots...")
    font_set_successfully = False
    preferred_font_name = "TH Sarabun New"

    try:
        ipython = get_ipython()
        in_colab_setup_fonts = ipython is not None and 'google.colab' in str(ipython)

        font_setup_logger.info("   Attempting to set font directly using findfont...")
        font_set_successfully = set_thai_font(preferred_font_name)

        if not font_set_successfully and in_colab_setup_fonts:
            font_setup_logger.info("\n   Preferred font not found. Attempting installation via apt-get (Colab)...")
            try:
                font_setup_logger.info("      Installing Thai fonts (fonts-thai-tlwg)... This might take a moment.")
                apt_update_process = subprocess.run(["apt-get", "update", "-qq"], check=False, capture_output=True, text=True, timeout=120)
                if apt_update_process.returncode != 0:
                    font_setup_logger.warning(f"      (Warning) apt-get update failed (Code: {apt_update_process.returncode}): {apt_update_process.stderr[:200]}...")
                apt_install_process = subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-thai-tlwg"], check=False, capture_output=True, text=True, timeout=180)
                if apt_install_process.returncode == 0:
                    font_setup_logger.info("      (Success) apt-get install fonts-thai-tlwg potentially completed.")
                    font_setup_logger.info("      Rebuilding Matplotlib font cache...")
                    try:
                        fm._load_fontmanager(try_read_cache=False) # type: ignore
                        font_setup_logger.info("      Font cache rebuilt. Attempting to set font again...")
                        font_set_successfully = set_thai_font(preferred_font_name)
                        if not font_set_successfully: font_set_successfully = set_thai_font("Loma")
                        if font_set_successfully: font_setup_logger.info("      (Success) Thai font set after installation and cache rebuild.")
                        else:
                            font_setup_logger.warning("      (Warning) Thai font still not set after installation. A manual Colab Runtime Restart might be needed.")
                            font_setup_logger.warning("      *****************************************************")
                            font_setup_logger.warning("      *** Please RESTART RUNTIME now for Matplotlib     ***")
                            font_setup_logger.warning("      *** to recognize the new fonts if plots fail.     ***")
                            font_setup_logger.warning("      *** (เมนู Runtime -> Restart runtime...)         ***")
                            font_setup_logger.warning("      *****************************************************")
                    except Exception as e_cache:
                        font_setup_logger.error(f"      (Error) Failed to rebuild font cache or set font after install: {e_cache}", exc_info=True)
                else:
                    font_setup_logger.warning(f"      (Warning) apt-get install failed (Code: {apt_install_process.returncode}): {apt_install_process.stderr[:200]}...")
            except subprocess.TimeoutExpired:
                font_setup_logger.error("      (Error) Timeout during apt-get font installation.")
            except Exception as e_generic_install:
                font_setup_logger.error(f"      (Error) General error during font installation attempt: {e_generic_install}", exc_info=True)

        if not font_set_successfully:
            fallback_fonts = ["Loma", "Garuda", "Norasi", "Kinnari", "Waree", "THSarabunNew"]
            font_setup_logger.info(f"\n   Trying fallbacks ({', '.join(fallback_fonts)})...")
            for fb_font in fallback_fonts:
                if set_thai_font(fb_font):
                    font_set_successfully = True
                    break
        if not font_set_successfully: font_setup_logger.critical("\n   (CRITICAL WARNING) Could not set any preferred Thai font. Plots WILL NOT render Thai characters correctly.")
        else: font_setup_logger.info("\n   (Info) Font setup process complete.")
    except Exception as e:
        font_setup_logger.error(f"   (Error) Critical error during font setup: {e}", exc_info=True)

# --- Data Loading Helper ---
def safe_load_csv_auto(file_path: str) -> pd.DataFrame | None:
    """
    Loads CSV or .csv.gz file using pandas, automatically handling gzip compression.
    """
    read_csv_kwargs = {"index_col": 0, "parse_dates": False, "low_memory": False}
    load_logger = logging.getLogger(f"{__name__}.safe_load_csv_auto")
    load_logger.info(f"      (safe_load) Attempting to load: {os.path.basename(file_path)}")

    if not isinstance(file_path, str) or not file_path:
        load_logger.error("         (Error) Invalid file path provided to safe_load_csv_auto.")
        return None
    if not os.path.exists(file_path):
        load_logger.error(f"         (Error) File not found: {file_path}")
        return None

    try:
        if file_path.lower().endswith(".gz"):
            load_logger.debug("         -> Detected .gz extension, using gzip.")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return pd.read_csv(f, **read_csv_kwargs)
        else:
            load_logger.debug("         -> No .gz extension, using standard pd.read_csv.")
            return pd.read_csv(file_path, **read_csv_kwargs)
    except pd.errors.EmptyDataError:
        load_logger.warning(f"         (Warning) File is empty: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        load_logger.error(f"         (Error) Failed to load file '{os.path.basename(file_path)}': {e}", exc_info=True)
        return None

# --- JSON Serialization Helper ---
def simple_converter(o):
    """
    Converts numpy/pandas types for JSON serialization.
    """
    json_logger = logging.getLogger(f"{__name__}.simple_converter")
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): # type: ignore
        if np.isnan(o): return None
        if np.isinf(o): return "Infinity" if o > 0 else "-Infinity"
        return float(o)
    if isinstance(o, pd.Timestamp): return o.isoformat()
    if isinstance(o, np.bool_): return bool(o)
    if pd.isna(o): return None
    if isinstance(o, (datetime.datetime, datetime.date)): return o.isoformat()
    try:
        if isinstance(o, (str, bool, list, dict, type(None))):
            json.dumps(o) # Test serialization
            return o
        str_representation = str(o)
        json_logger.debug(f"Type {type(o)} not directly serializable. Converting to string: '{str_representation[:100]}...'")
        return str_representation
    except TypeError:
        str_representation_on_error = str(o)
        json_logger.warning(f"TypeError for type {type(o)}. Using str(): '{str_representation_on_error[:100]}...'")
        return str_representation_on_error
    except Exception as e:
        str_representation_on_general_error = str(o)
        json_logger.error(f"Unexpected error for type {type(o)}: {e}. Using str(): '{str_representation_on_general_error[:100]}...'", exc_info=True)
        return str_representation_on_general_error

# --- Configuration Loading Helper (DEPRECATED by load_config_from_yaml in Part 3) ---
def load_app_config(config_path="config_main.json") -> dict:
    """
    DEPRECATED: Loads application configuration from a JSON file.
    Use load_config_from_yaml and StrategyConfig instead for new configurations.
    """
    dep_logger = logging.getLogger(f"{__name__}.load_app_config")
    dep_logger.warning("Function 'load_app_config' is DEPRECATED. Use 'load_config_from_yaml' with StrategyConfig instead.")
    try:
        script_dir = ""
        try: script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: script_dir = os.getcwd(); dep_logger.debug(f"   (Config) __file__ not defined, using CWD: {script_dir} for config lookup.")
        potential_path = os.path.join(script_dir, config_path)
        actual_config_path = None
        if os.path.exists(potential_path): actual_config_path = potential_path
        elif os.path.exists(config_path): actual_config_path = config_path; dep_logger.debug(f"   (Config) Config not found at '{potential_path}', trying '{config_path}' in CWD.")
        if actual_config_path is None: raise FileNotFoundError(f"Configuration file '{config_path}' not found in script directory ('{script_dir}') or CWD.")
        with open(actual_config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
        dep_logger.info(f"   (Config) Successfully loaded configuration from: {actual_config_path}")
        return config_data
    except FileNotFoundError: dep_logger.error(f"   (Config Error) Configuration file '{config_path}' not found. Using default script values."); return {}
    except json.JSONDecodeError: dep_logger.error(f"   (Config Error) Error decoding JSON from configuration file: {config_path}. Using default script values."); return {}
    except Exception as e: dep_logger.error(f"   (Config Error) Failed to load configuration from '{config_path}': {e}", exc_info=True); return {}

# --- Datetime Setting Helper ---
def safe_set_datetime(df: pd.DataFrame, idx, col: str, val):
    """
    Safely assigns datetime value to DataFrame, robustly ensuring column dtype is datetime64[ns].
    """
    dt_logger = logging.getLogger(f"{__name__}.safe_set_datetime")
    try:
        dt_value_orig = pd.to_datetime(val, errors='coerce')
        dt_logger.debug(f"   [safe_set_datetime] Input val='{val}', Original dt_value='{dt_value_orig}' (type: {type(dt_value_orig)}, tz: {dt_value_orig.tzinfo if isinstance(dt_value_orig, pd.Timestamp) else 'N/A'}) for col='{col}', idx='{idx}'")
        dt_value = dt_value_orig
        if isinstance(dt_value_orig, pd.Timestamp) and dt_value_orig.tzinfo is not None:
            dt_value = dt_value_orig.tz_localize(None)
            dt_logger.debug(f"      Converted timezone-aware Timestamp to naive: '{dt_value}'")
        elif pd.isna(dt_value_orig): dt_value = pd.NaT; dt_logger.debug(f"      dt_value is NaT.")
        else: dt_logger.debug(f"      dt_value is already timezone-naive or not a Timestamp: '{dt_value}' (type: {type(dt_value)})")

        if col not in df.columns:
            dt_logger.info(f"   [safe_set_datetime] Column '{col}' not found. Creating with dtype 'datetime64[ns]'.")
            df[col] = pd.Series([pd.NaT]*len(df), index=df.index, dtype='datetime64[ns]') # type: ignore
            dt_logger.debug(f"      Column '{col}' created. Dtype after creation: {df[col].dtype}")
        elif df[col].dtype != 'datetime64[ns]':
            dt_logger.warning(f"   [safe_set_datetime] Column '{col}' has dtype '{df[col].dtype}'. Attempting conversion to 'datetime64[ns]'.")
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').astype('datetime64[ns]')
                dt_logger.info(f"      Column '{col}' successfully converted to 'datetime64[ns]'. Dtype after conversion: {df[col].dtype}")
            except Exception as e_conv_col:
                dt_logger.error(f"   [safe_set_datetime] CRITICAL - Could not convert column '{col}' to datetime64[ns] ({e_conv_col}). Re-creating column. THIS MAY LOSE DATA IN THE COLUMN.")
                df[col] = pd.Series([pd.NaT]*len(df), index=df.index, dtype='datetime64[ns]') # type: ignore
                dt_logger.debug(f"      Column '{col}' re-created due to conversion failure. Dtype after re-creation: {df[col].dtype}")

        if idx in df.index:
            dtype_before_assign = df.loc[idx, col].dtype if col in df.columns and idx in df.index and hasattr(df.loc[idx, col], 'dtype') else 'N/A (Scalar or New)'
            dt_logger.debug(f"      Assigning (naive) '{dt_value}' to '{col}' at index {idx}. Dtype of cell before assign: {dtype_before_assign}. Dtype of column '{col}': {df[col].dtype}")
            df.loc[idx, col] = dt_value
            dtype_after_assign_cell = df.loc[idx, col].dtype if hasattr(df.loc[idx, col], 'dtype') else type(df.loc[idx, col])
            dt_logger.debug(f"      Assigned. Dtype of cell after assign: {dtype_after_assign_cell}. Dtype of column '{col}' after assign: {df[col].dtype}")
            if df[col].dtype != 'datetime64[ns]':
                dt_logger.error(f"   [safe_set_datetime] UNEXPECTED - Column '{col}' dtype is '{df[col].dtype}' AFTER assigning a naive datetime. This should not happen if pre-allocation was correct.")
                try:
                    df[col] = df[col].astype('datetime64[ns]')
                    dt_logger.warning(f"         [safe_set_datetime] Column '{col}' dtype after UNEXPECTED force astype: {df[col].dtype}")
                except Exception as e_final_astype:
                    dt_logger.critical(f"         [safe_set_datetime] CRITICAL - Failed to force astype on column '{col}' after unexpected dtype change: {e_final_astype}")
        else:
            dt_logger.warning(f"   [safe_set_datetime] Index '{idx}' not found in DataFrame. Cannot set value for column '{col}'.")
    except Exception as e:
        dt_logger.error(f"   (Error) safe_set_datetime: Outer exception for val='{val}', col='{col}', idx='{idx}': {e}", exc_info=True)
        try:
            if idx in df.index:
                if col not in df.columns or df[col].dtype != 'datetime64[ns]':
                    df[col] = pd.Series([pd.NaT]*len(df), index=df.index, dtype='datetime64[ns]') # type: ignore
                df.loc[idx, col] = pd.NaT
                dt_logger.debug(f"      [safe_set_datetime] Fallback (Outer Exception): Assigned NaT to '{col}' at index {idx}. Column dtype: {df[col].dtype}")
            else:
                dt_logger.warning(f"   [safe_set_datetime] Index '{idx}' not found during fallback (Outer Exception) for column '{col}'.")
        except Exception as e_fallback:
            dt_logger.error(f"   (Error) safe_set_datetime: Failed to assign NaT as fallback (Outer Exception) for '{col}' at index {idx}: {e_fallback}")

logger.info("Part 4 (Original Part 3): Helper Functions (Setup, Utils, Font, Config) Loaded.")
# === END OF PART 4/15 ===
# === START OF PART 5/15 ===
# ==============================================================================
# === PART 5: Data Loading & Initial Preparation (v4.9.0 - Enterprise Refactor) ===
# ==============================================================================
# <<< MODIFIED: Enterprise Refactor - Loggers made more specific. >>>
# <<< MAX_NAT_RATIO_THRESHOLD will be accessed via StrategyConfig later. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.3] Removed global MAX_NAT_RATIO_THRESHOLD; value is now sourced from StrategyConfig in prepare_datetime. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.4] Integrated MAX_NAT_RATIO_THRESHOLD into StrategyConfig and updated usage. >>>

import logging # Already imported
import os # Already imported
import sys # Already imported
import pandas as pd # Already imported
import numpy as np # Already imported
import warnings # Already imported
import traceback # Already imported
import datetime # Standard datetime import # Already imported
import gc # For memory management

# --- Data Loading Function ---
def load_data(file_path: str, timeframe_str: str = "", price_jump_threshold: float = 0.10, nan_threshold: float = 0.05, dtypes: dict | None = None) -> pd.DataFrame | None:
    """
    Loads data from a CSV file, performs basic validation and data quality checks.
    """
    load_data_logger = logging.getLogger(f"{__name__}.load_data")
    load_data_logger.info(f"(Loading) กำลังโหลดข้อมูล {timeframe_str} จาก: {file_path}")
    if not os.path.exists(file_path):
        load_data_logger.critical(f"(Error) ไม่พบไฟล์: {file_path}")
        sys.exit(f"ออก: ไม่พบไฟล์ข้อมูล {timeframe_str} ที่ {file_path}")

    try:
        try:
            df_pd = pd.read_csv(file_path, low_memory=False, dtype=dtypes)
            load_data_logger.info(f"   ไฟล์ดิบ {timeframe_str}: {df_pd.shape[0]} แถว")
        except pd.errors.ParserError as e_parse:
            load_data_logger.critical(f"(Error) ไม่สามารถ Parse ไฟล์ CSV '{file_path}': {e_parse}")
            sys.exit(f"ออก: ปัญหาการ Parse ไฟล์ CSV {timeframe_str}")
        except Exception as e_read:
            load_data_logger.critical(f"(Error) ไม่สามารถอ่านไฟล์ CSV '{file_path}': {e_read}", exc_info=True)
            sys.exit(f"ออก: ปัญหาการอ่านไฟล์ CSV {timeframe_str}")

        required_cols_base = ["Date", "Timestamp", "Open", "High", "Low", "Close"]
        required_cols_check = list(dtypes.keys()) if dtypes else required_cols_base
        required_cols_check = sorted(list(set(required_cols_check + required_cols_base)))
        missing_req = [col for col in required_cols_check if col not in df_pd.columns]
        if missing_req:
            load_data_logger.critical(f"(Error) ขาดคอลัมน์: {missing_req} ใน {file_path}")
            sys.exit(f"ออก: ขาดคอลัมน์ที่จำเป็นในข้อมูล {timeframe_str}")

        price_cols = ["Open", "High", "Low", "Close"]
        load_data_logger.debug(f"   Converting price columns {price_cols} to numeric (if not already specified in dtypes)...")
        for col in price_cols:
            if dtypes is None or col not in dtypes or not pd.api.types.is_numeric_dtype(df_pd[col].dtype):
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')

        load_data_logger.debug("   [Data Quality] Checking for invalid prices (<= 0)...")
        for col in price_cols:
            invalid_prices = df_pd[pd.notna(df_pd[col]) & (df_pd[col] <= 0)]
            if not invalid_prices.empty:
                load_data_logger.warning(f"   (Warning) พบราคาที่ผิดปกติ (<= 0) ในคอลัมน์ '{col}' จำนวน {len(invalid_prices)} แถว. แถวตัวอย่าง:\n{invalid_prices.head()}")

        load_data_logger.debug("   [Data Quality] Checking High >= Low consistency...")
        invalid_hl = df_pd[pd.notna(df_pd['High']) & pd.notna(df_pd['Low']) & (df_pd['High'] < df_pd['Low'])]
        if not invalid_hl.empty:
            load_data_logger.warning(f"   (Warning) พบราคา High < Low จำนวน {len(invalid_hl)} แถว. แถวตัวอย่าง:\n{invalid_hl.head()}")

        load_data_logger.info("   [Data Quality] ตรวจสอบ % NaN ในคอลัมน์ราคา...")
        nan_report = df_pd[price_cols].isnull().mean()
        load_data_logger.info(f"      NaN Percentage:\n{nan_report.round(4)}")
        high_nan_cols = nan_report[nan_report > nan_threshold].index.tolist()
        if high_nan_cols:
            load_data_logger.warning(f"   (Warning) คอลัมน์ {high_nan_cols} มี NaN เกินเกณฑ์ ({nan_threshold:.1%}).")

        initial_rows = df_pd.shape[0]
        df_pd.dropna(subset=price_cols, inplace=True)
        rows_dropped_nan = initial_rows - df_pd.shape[0]
        if rows_dropped_nan > 0:
            load_data_logger.info(f"   ลบ {rows_dropped_nan} แถวที่มีราคาเป็น NaN.")

        load_data_logger.info("   [Data Quality] ตรวจสอบ Duplicates (Date & Timestamp)...")
        duplicate_cols = ["Date", "Timestamp"]
        if all(col in df_pd.columns for col in duplicate_cols):
            num_duplicates = df_pd.duplicated(subset=duplicate_cols, keep=False).sum()
            if num_duplicates > 0:
                load_data_logger.warning(f"   (Warning) พบ {num_duplicates} แถวที่มี Date & Timestamp ซ้ำกัน. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
                df_pd.drop_duplicates(subset=duplicate_cols, keep='first', inplace=True)
                load_data_logger.info(f"      ขนาดข้อมูลหลังลบ Duplicates: {df_pd.shape[0]} แถว.")
            else:
                load_data_logger.debug("      ไม่พบ Duplicates (Date & Timestamp).")
        else:
            load_data_logger.warning(f"   (Warning) ขาดคอลัมน์ {duplicate_cols} สำหรับตรวจสอบ Duplicates.")

        load_data_logger.info(f"   [Data Quality] ตรวจสอบ Price Jumps (Threshold > {price_jump_threshold:.1%})...")
        if 'Close' in df_pd.columns and len(df_pd) > 1:
            df_pd['Close'] = pd.to_numeric(df_pd['Close'], errors='coerce')
            close_numeric = df_pd['Close'].dropna()
            if len(close_numeric) > 1:
                price_pct_change = close_numeric.pct_change().abs()
                large_jumps = price_pct_change[price_pct_change > price_jump_threshold]
                if not large_jumps.empty:
                    load_data_logger.warning(f"   (Warning) พบ {len(large_jumps)} แท่งที่มีการเปลี่ยนแปลงราคา Close เกิน {price_jump_threshold:.1%}:")
                    example_jumps = large_jumps.head()
                    load_data_logger.warning(f"      ตัวอย่าง Index และ % Change:\n{example_jumps.round(4).to_string()}")
                else:
                    load_data_logger.debug("      ไม่พบ Price Jumps ที่ผิดปกติ.")
                del close_numeric, price_pct_change, large_jumps
                gc.collect()
            else:
                load_data_logger.debug("      ข้ามการตรวจสอบ Price Jumps (ข้อมูล Close ไม่พอหลัง dropna).")
        else:
            load_data_logger.debug("      ข้ามการตรวจสอบ Price Jumps (ไม่มีข้อมูล Close หรือมีน้อยกว่า 2 แถว).")

        if df_pd.empty:
            load_data_logger.warning(f"   (Warning) DataFrame ว่างเปล่าหลังจากลบราคา NaN และ Duplicates ({timeframe_str}).")

        load_data_logger.info(f"(Success) โหลดและตรวจสอบข้อมูล {timeframe_str} สำเร็จ: {df_pd.shape[0]} แถว")
        return df_pd

    except SystemExit as se:
        raise se
    except Exception as e:
        load_data_logger.critical(f"(Error) ไม่สามารถโหลดข้อมูล {timeframe_str}: {e}\n{traceback.format_exc()}", exc_info=True)
        sys.exit(f"ออก: ข้อผิดพลาดร้ายแรงในการโหลดข้อมูล {timeframe_str}")
    return None

# --- Datetime Helper Functions ---
def preview_datetime_format(df: pd.DataFrame, n: int = 5):
    """Displays a preview of the Date + Timestamp string format before conversion."""
    preview_logger = logging.getLogger(f"{__name__}.preview_datetime_format")
    if df is None or df.empty or "Date" not in df.columns or "Timestamp" not in df.columns:
        preview_logger.warning("   [Preview] Cannot preview: DataFrame is empty or missing Date/Timestamp columns.")
        return
    preview_logger.info(f"   [Preview] First {n} Date + Timestamp format examples:")
    try:
        preview_df = df.head(n).copy()
        preview_df["Date"] = preview_df["Date"].astype(str).str.strip()
        preview_df["Timestamp"] = (
            preview_df["Timestamp"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        preview = preview_df.apply(lambda row: f"{row['Date']} {row['Timestamp']}", axis=1)
        preview_logger.info("\n" + preview.to_string(index=False))
        del preview_df, preview
        gc.collect()
    except Exception as e:
        preview_logger.error(f"   [Preview] Error during preview generation: {e}", exc_info=True)

def parse_datetime_safely(datetime_str_series: pd.Series) -> pd.Series:
    """
    Attempts to parse a Series of datetime strings into datetime objects using multiple formats.
    """
    parser_logger = logging.getLogger(f"{__name__}.parse_datetime_safely")
    if not isinstance(datetime_str_series, pd.Series):
        parser_logger.error("Input must be a pandas Series.")
        raise TypeError("Input must be a pandas Series.")
    if datetime_str_series.empty:
        parser_logger.debug("      [Parser] Input series is empty, returning empty series.")
        return datetime_str_series.astype("datetime64[ns]")

    parser_logger.info("      [Parser] Attempting to parse date/time strings...")
    common_formats = [
        "%Y%m%d %H:%M:%S", "%Y%m%d%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    parsed_results = pd.Series(pd.NaT, index=datetime_str_series.index, dtype="datetime64[ns]")
    remaining_indices = datetime_str_series.index.copy()
    series_to_parse = datetime_str_series.copy()

    for fmt in common_formats:
        if remaining_indices.empty:
            parser_logger.debug("      [Parser] All strings parsed.")
            break
        parser_logger.debug(f"      [Parser] Trying format: '{fmt}' for {len(remaining_indices)} values...")
        try:
            try_parse = pd.to_datetime(
                series_to_parse.loc[remaining_indices], format=fmt, errors='coerce'
            )
            successful_mask_this_attempt = try_parse.notna()
            successful_indices_this_attempt = remaining_indices[successful_mask_this_attempt]
            if not successful_indices_this_attempt.empty:
                parsed_results.loc[successful_indices_this_attempt] = try_parse[successful_mask_this_attempt]
                remaining_indices = remaining_indices.difference(successful_indices_this_attempt)
                parser_logger.info(
                    f"      [Parser] (Success) Format '{fmt}' matched: {len(successful_indices_this_attempt)}. Remaining: {len(remaining_indices)}"
                )
            del try_parse, successful_mask_this_attempt, successful_indices_this_attempt
            gc.collect()
        except ValueError as ve:
            if not remaining_indices.empty:
                first_failed_idx = remaining_indices[0]
                first_failed_str = series_to_parse.get(first_failed_idx, 'N/A')
                parser_logger.warning(f"      [Parser] Invalid format '{fmt}' for string like '{first_failed_str}' (ValueError: {ve}). Trying next format.")
            else:
                parser_logger.warning(f"      [Parser] ValueError with format '{fmt}' but no remaining indices to sample from (ValueError: {ve}).")
        except Exception as e_fmt:
            parser_logger.warning(f"         -> General error while trying format '{fmt}': {e_fmt}", exc_info=True)

    if not remaining_indices.empty:
        parser_logger.info(f"      [Parser] Trying general parser for {len(remaining_indices)} remaining values...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                try_general = pd.to_datetime(series_to_parse.loc[remaining_indices], errors='coerce')
            successful_mask_general = try_general.notna()
            successful_indices_general = remaining_indices[successful_mask_general]
            if not successful_indices_general.empty:
                parsed_results.loc[successful_indices_general] = try_general[successful_mask_general]
                remaining_indices = remaining_indices.difference(successful_indices_general)
                parser_logger.info(f"         -> (Success) General parser matched: {len(successful_indices_general)}. Remaining: {len(remaining_indices)}")
            del try_general, successful_mask_general, successful_indices_general
            gc.collect()
        except Exception as e_gen:
            parser_logger.warning(f"         -> General parser error: {e_gen}", exc_info=True)

    final_nat_count = parsed_results.isna().sum()
    if final_nat_count > 0:
        parser_logger.warning(f"      [Parser] Could not parse {final_nat_count} date/time strings.")
        failed_strings_log = series_to_parse[parsed_results.isna()].head(5)
        parser_logger.warning(f"         Example failed strings:\n{failed_strings_log.to_string()}")
    parser_logger.info("      [Parser] (Finished) Date/time parsing complete.")
    del series_to_parse, remaining_indices
    gc.collect()
    return parsed_results

# <<< MODIFIED: [Patch AI Studio v4.9.4] Integrated MAX_NAT_RATIO_THRESHOLD into StrategyConfig and updated usage. >>>
def prepare_datetime(df_pd: pd.DataFrame, timeframe_str: str = "", config: Optional['StrategyConfig'] = None) -> pd.DataFrame: # type: ignore
    """
    Prepares the DatetimeIndex for the DataFrame, handling Buddhist Era conversion
    and NaT values. Sets the prepared datetime as the DataFrame index.
    Uses max_nat_ratio_threshold from the config object.
    """
    prep_dt_logger = logging.getLogger(f"{__name__}.prepare_datetime")
    prep_dt_logger.info(f"(Processing) กำลังเตรียม Datetime Index ({timeframe_str})...")
    if not isinstance(df_pd, pd.DataFrame):
        prep_dt_logger.error("Input must be a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")
    if df_pd.empty:
        prep_dt_logger.warning(f"   (Warning) prepare_datetime: DataFrame ว่างเปล่า ({timeframe_str}). Returning empty DataFrame.")
        return df_pd.copy()

    max_nat_ratio_from_config = 0.05 # Default fallback
    if config is not None and hasattr(config, 'max_nat_ratio_threshold'):
        max_nat_ratio_from_config = config.max_nat_ratio_threshold
        prep_dt_logger.debug(f"   Using max_nat_ratio_threshold from config: {max_nat_ratio_from_config}")
    else:
        prep_dt_logger.warning(f"   StrategyConfig not provided or missing 'max_nat_ratio_threshold'. Using default: {max_nat_ratio_from_config}")


    try:
        if "Date" not in df_pd.columns or "Timestamp" not in df_pd.columns:
            prep_dt_logger.critical(f"(Error) ขาดคอลัมน์ 'Date'/'Timestamp' ใน {timeframe_str}.")
            sys.exit(f"ออก ({timeframe_str}): ขาดคอลัมน์ Date/Timestamp ที่จำเป็นสำหรับการเตรียม Datetime.")

        preview_datetime_format(df_pd)

        date_str_series = df_pd["Date"].astype(str).str.strip()
        ts_str_series = (
            df_pd["Timestamp"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

        prep_dt_logger.info("      [Converter] กำลังตรวจสอบและแปลงปี พ.ศ. เป็น ค.ศ. (ถ้าจำเป็น)...")
        converted_date_str_series = date_str_series.copy()
        if not date_str_series.empty:
            potential_be = False
            sample_size = min(len(date_str_series), 100)
            try:
                if date_str_series.index.is_unique:
                    sampled_dates = date_str_series.sample(sample_size, random_state=42)
                else:
                    sampled_dates = date_str_series.drop_duplicates().sample(min(sample_size, date_str_series.nunique()), random_state=42)
            except Exception as e_sample:
                prep_dt_logger.warning(f"      [Converter] Warning: Sampling failed ({e_sample}). Proceeding without sampling check.")
                sampled_dates = date_str_series

            for date_str_sample in sampled_dates:
                if isinstance(date_str_sample, str):
                    year_part_str = None
                    if len(date_str_sample) >= 4:
                        if date_str_sample[:4].isdigit(): year_part_str = date_str_sample[:4]
                    if year_part_str:
                        try:
                            year_part = int(year_part_str)
                            current_ce_year = datetime.datetime.now().year
                            if year_part > current_ce_year + 100:
                                potential_be = True
                                prep_dt_logger.debug(f"      [Converter] Potential BE year detected: {year_part} in '{date_str_sample}'")
                                break
                        except ValueError: continue
            del sampled_dates

            if potential_be:
                prep_dt_logger.info("      [Converter] ตรวจพบปีที่อาจเป็น พ.ศ. (> current_year + 100). พยายามแปลงเป็น ค.ศ. (-543)...")
                def convert_be_year(date_str_conv):
                    if isinstance(date_str_conv, str) and len(date_str_conv) >= 4:
                        year_part_str_conv_inner = date_str_conv[:4]
                        if year_part_str_conv_inner.isdigit():
                            try:
                                year_be_conv = int(year_part_str_conv_inner)
                                current_ce_year_conv_inner = datetime.datetime.now().year
                                if year_be_conv > current_ce_year_conv_inner + 100:
                                    year_ce_conv = year_be_conv - 543
                                    return str(year_ce_conv) + date_str_conv[4:]
                            except ValueError:
                                return date_str_conv
                    return date_str_conv
                converted_date_str_series = date_str_series.apply(convert_be_year)
                if not converted_date_str_series.equals(date_str_series):
                    prep_dt_logger.info("      [Converter] (Success) แปลงปี พ.ศ. เป็น ค.ศ. สำเร็จ.")
                    diff_mask = date_str_series != converted_date_str_series
                    prep_dt_logger.info(f"         ตัวอย่างก่อนแปลง:\n{date_str_series[diff_mask].head(3).to_string(index=False)}")
                    prep_dt_logger.info(f"         ตัวอย่างหลังแปลง:\n{converted_date_str_series[diff_mask].head(3).to_string(index=False)}")
                    del diff_mask
                else:
                    prep_dt_logger.info("      [Converter] ไม่พบปีที่น่าจะเป็น พ.ศ. ที่ต้องแปลง (หรือข้อมูลน้อยเกินไป).")
            else:
                prep_dt_logger.info("      [Converter] ไม่พบปีที่น่าจะเป็น พ.ศ. (หรือข้อมูลน้อยเกินไป).")

        prep_dt_logger.debug("      Combining Date and Timestamp strings...")
        datetime_combined_str = converted_date_str_series + " " + ts_str_series
        df_pd["datetime_original"] = parse_datetime_safely(datetime_combined_str)
        del date_str_series, ts_str_series, converted_date_str_series
        gc.collect()

        nat_count = df_pd["datetime_original"].isna().sum()
        if nat_count > 0:
            nat_ratio = nat_count / len(df_pd) if len(df_pd) > 0 else 0
            prep_dt_logger.warning(f"   (Warning) พบค่า NaT {nat_count} ({nat_ratio:.1%}) ใน {timeframe_str} หลังการ parse.")

            if nat_ratio == 1.0:
                failed_strings = datetime_combined_str[df_pd["datetime_original"].isna()]
                prep_dt_logger.critical(f"   (Error) พบค่า NaT 100% ใน {timeframe_str}. ไม่สามารถดำเนินการต่อได้. ตัวอย่าง: {failed_strings.iloc[0] if not failed_strings.empty else 'N/A'}")
                sys.exit(f"   ออก ({timeframe_str}): ข้อมูล date/time ทั้งหมดไม่สามารถ parse ได้.")
            elif nat_ratio >= max_nat_ratio_from_config:
                prep_dt_logger.warning(f"   (Warning) สัดส่วน NaT ({nat_ratio:.1%}) เกินเกณฑ์ ({max_nat_ratio_from_config:.1%}) แต่ไม่ใช่ 100%.")
                prep_dt_logger.warning(f"   (Warning) Fallback: ลบ {nat_count} แถว NaT และดำเนินการต่อ...")
                df_pd.dropna(subset=["datetime_original"], inplace=True)
                if df_pd.empty:
                    prep_dt_logger.critical(f"   (Error) ข้อมูล {timeframe_str} ทั้งหมดเป็น NaT หรือใช้ไม่ได้หลัง fallback (และ DataFrame ว่างเปล่า).")
                    sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังลบ NaT เกินเกณฑ์.")
                prep_dt_logger.info(f"   (Success) ดำเนินการต่อด้วย {len(df_pd)} แถวที่เหลือ ({timeframe_str}).")
            else:
                prep_dt_logger.info(f"   กำลังลบ {nat_count} แถว NaT (ต่ำกว่าเกณฑ์).")
                df_pd.dropna(subset=["datetime_original"], inplace=True)
                if df_pd.empty:
                    prep_dt_logger.critical(f"   (Error) ข้อมูล {timeframe_str} ว่างเปล่าหลังลบ NaT จำนวนเล็กน้อย.")
                    sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังลบ NaT.")
        else:
            prep_dt_logger.debug(f"   ไม่พบค่า NaT ใน {timeframe_str} หลังการ parse.")
        del datetime_combined_str
        gc.collect()

        if "datetime_original" in df_pd.columns:
            df_pd["datetime_original"] = pd.to_datetime(df_pd["datetime_original"], errors='coerce')
            df_pd = df_pd[~df_pd["datetime_original"].isna()]
            if df_pd.empty:
                prep_dt_logger.critical(f"   (Error) ข้อมูล {timeframe_str} ว่างเปล่าหลังแปลง datetime_original และลบ NaT (ก่อน set_index).")
                sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังการเตรียม datetime.")
            df_pd.set_index(pd.DatetimeIndex(df_pd["datetime_original"]), inplace=True)
        else:
            prep_dt_logger.critical(f"   (Error) คอลัมน์ 'datetime_original' หายไปก่อนการตั้งค่า Index ({timeframe_str}).")
            sys.exit(f"   ออก ({timeframe_str}): ขาดคอลัมน์ 'datetime_original'.")

        df_pd.sort_index(inplace=True)

        if df_pd.index.has_duplicates:
            initial_rows_dedup = df_pd.shape[0]
            prep_dt_logger.warning(f"   (Warning) พบ Index ซ้ำ {df_pd.index.duplicated().sum()} รายการ. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
            df_pd = df_pd[~df_pd.index.duplicated(keep='first')]
            prep_dt_logger.info(f"   แก้ไข index ซ้ำ: ลบ {initial_rows_dedup - df_pd.shape[0]} แถว.")

        prep_dt_logger.debug("   Checking for non-monotonic index (time reversals)...")
        time_diffs = df_pd.index.to_series().diff()
        negative_diffs = time_diffs[time_diffs < pd.Timedelta(0)]
        if not negative_diffs.empty:
            prep_dt_logger.critical(f"   (CRITICAL WARNING) พบเวลาย้อนกลับใน Index ของ {timeframe_str} หลังการเรียงลำดับ!")
            prep_dt_logger.critical(f"      จำนวน: {len(negative_diffs)}")
            prep_dt_logger.critical(f"      ตัวอย่าง Index ที่มีปัญหา:\n{negative_diffs.head()}")
            sys.exit(f"   ออก ({timeframe_str}): พบเวลาย้อนกลับในข้อมูล.")
        else:
            prep_dt_logger.debug("      Index is monotonic increasing.")
        del time_diffs, negative_diffs

        prep_dt_logger.info(f"(Success) เตรียม Datetime index ({timeframe_str}) สำเร็จ. Shape: {df_pd.shape}")
        return df_pd

    except SystemExit as se:
        raise se
    except ValueError as ve:
        prep_dt_logger.critical(f"   (Error) prepare_datetime: ValueError: {ve}", exc_info=True)
        sys.exit(f"   ออก ({timeframe_str}): ปัญหาข้อมูล Date/time.")
    except Exception as e:
        prep_dt_logger.critical(f"(Error) ข้อผิดพลาดร้ายแรงใน prepare_datetime ({timeframe_str}): {e}", exc_info=True)
        sys.exit(f"   ออก ({timeframe_str}): ข้อผิดพลาดร้ายแรงในการเตรียม datetime.")
    return df_pd
# <<< END OF MODIFIED [Patch AI Studio v4.9.4] >>>

logger.info("Part 5 (Original Part 4): Data Loading & Initial Preparation Functions Loaded.")
# === END OF PART 5/15 ===
# === START OF PART 6/15 ===
# ==============================================================================
# === PART 6: Feature Engineering & Indicator Calculation (v4.9.15 - Signals Use Config Defaults) ===
# ==============================================================================
# <<< MODIFIED: calculate_m1_entry_signals now correctly uses default thresholds from StrategyConfig. >>>
# <<< MODIFIED: [Patch] Ensured DataFrame boolean checks use .empty in calculate_m15_trend_zone. >>>

import logging  # Already imported
import pandas as pd  # Already imported
import numpy as np  # Already imported
import ta  # Assumes 'ta' is imported and available (checked in Part 1)
from sklearn.cluster import KMeans  # For context column calculation
from sklearn.preprocessing import StandardScaler  # For context column calculation
import gc  # For memory management

# --- Feature Engineering Constants are NOW REMOVED FROM HERE ---
# They are accessed via the `config: StrategyConfig` object passed to relevant functions.

# --- Indicator Calculation Functions (Unchanged in this refactor step, already robust) ---
def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    ema_logger = logging.getLogger(f"{__name__}.ema")
    if not isinstance(series, pd.Series):
        ema_logger.error(f"Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        ema_logger.debug("Input series is empty, returning empty series.")
        return pd.Series(dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty: # pragma: no cover
        ema_logger.warning("Series contains only NaN/Inf values or is empty after cleaning.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        min_p = max(1, min(period, len(series_numeric))) # Ensure min_periods is at least 1 and not more than series length
        ema_calculated = series_numeric.ewm(span=period, adjust=False, min_periods=min_p).mean()
        ema_result = ema_calculated.reindex(series.index) # Reindex to original to keep NaNs where they were
        del series_numeric, ema_calculated
        gc.collect()
        return ema_result.astype('float32')
    except Exception as e: # pragma: no cover
        ema_logger.error(f"EMA calculation failed for period {period}: {e}", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype='float32')

def sma(series: pd.Series, period: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    sma_logger = logging.getLogger(f"{__name__}.sma")
    if not isinstance(series, pd.Series):
        sma_logger.error(f"Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        sma_logger.debug("Input series is empty, returning empty series.")
        return pd.Series(dtype='float32')
    if not isinstance(period, int) or period <= 0: # pragma: no cover
        sma_logger.error(f"Invalid period ({period}). Must be a positive integer.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0) # Fill NaNs with 0 for SMA
    if series_numeric.isnull().all(): # pragma: no cover
        sma_logger.warning("Series contains only NaN values after numeric conversion and fill.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        min_p = max(1, min(period, len(series_numeric)))
        sma_result = series_numeric.rolling(window=period, min_periods=min_p).mean()
        sma_final = sma_result.reindex(series.index) # Reindex to original
        del series_numeric, sma_result
        gc.collect()
        return sma_final.astype('float32')
    except Exception as e: # pragma: no cover
        sma_logger.error(f"SMA calculation failed for period {period}: {e}", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype='float32')

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index."""
    rsi_logger = logging.getLogger(f"{__name__}.rsi")
    if not isinstance(series, pd.Series):
        rsi_logger.error(f"Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        rsi_logger.debug("Input series is empty, returning empty series.")
        return pd.Series(dtype='float32')
    if 'ta' not in globals() or ta is None: # pragma: no cover
        rsi_logger.error("   (Error) RSI calculation failed: 'ta' library not loaded.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty or len(series_numeric) < period: # pragma: no cover
        rsi_logger.warning(f"   (Warning) RSI calculation skipped: Not enough valid data points ({len(series_numeric)} < {period}).")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=series_numeric, window=period, fillna=False)  # type: ignore
        rsi_values = rsi_indicator.rsi()
        rsi_final = rsi_values.reindex(series.index).ffill() # Forward fill to propagate last valid RSI
        del series_numeric, rsi_indicator, rsi_values
        gc.collect()
        return rsi_final.astype('float32')
    except Exception as e: # pragma: no cover
        rsi_logger.error(f"   (Error) RSI calculation error for period {period}: {e}.", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype='float32')

def atr(df_in: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculates Average True Range and adds ATR_{period} and ATR_{period}_Shifted columns."""
    atr_logger = logging.getLogger(f"{__name__}.atr")
    if not isinstance(df_in, pd.DataFrame):
        atr_logger.error(f"Input must be a pandas DataFrame, got {type(df_in)}")
        raise TypeError("Input must be a pandas DataFrame.")

    atr_col_name = f"ATR_{period}"
    atr_shifted_col_name = f"ATR_{period}_Shifted"
    df_result = df_in.copy()

    if df_in.empty:
        atr_logger.debug("Input DataFrame is empty. Returning with NaN ATR columns.")
        df_result[atr_col_name] = np.nan
        df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32')
        df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32')
        return df_result

    df_temp = df_in.copy()
    required_price_cols = ["High", "Low", "Close"]
    if not all(col in df_temp.columns for col in required_price_cols): # pragma: no cover
        atr_logger.warning(f"   (Warning) ATR calculation skipped: Missing columns {required_price_cols}.")
        df_result[atr_col_name] = np.nan
        df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32')
        df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32')
        return df_result

    for col in required_price_cols: # Convert to numeric and handle Inf/NaN
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_temp.dropna(subset=required_price_cols, inplace=True)

    if df_temp.empty or len(df_temp) < period: # pragma: no cover
        atr_logger.warning(f"   (Warning) ATR calculation skipped: Not enough valid data after dropna (need >= {period}).")
        df_result[atr_col_name] = np.nan
        df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32')
        df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32')
        return df_result

    atr_series = None
    if 'ta' in globals() and ta is not None:
        try:
            atr_indicator = ta.volatility.AverageTrueRange(high=df_temp['High'], low=df_temp['Low'], close=df_temp['Close'], window=period, fillna=False)  # type: ignore
            atr_series = atr_indicator.average_true_range()
            del atr_indicator
        except Exception as e_ta_atr: # pragma: no cover
            atr_logger.warning(f"   (Warning) TA library ATR calculation failed: {e_ta_atr}. Falling back to manual.")
            atr_series = None # Ensure it's None to trigger manual calc

    if atr_series is None: # Fallback to manual calculation
        try:
            df_temp['H-L'] = df_temp['High'] - df_temp['Low']
            df_temp['H-PC'] = abs(df_temp['High'] - df_temp['Close'].shift(1))
            df_temp['L-PC'] = abs(df_temp['Low'] - df_temp['Close'].shift(1))
            df_temp['TR'] = df_temp[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            if not df_temp.empty and len(df_temp) > 0: # Check if df_temp is not empty
                first_valid_index = df_temp.index[0]
                if first_valid_index in df_temp.index: # Ensure index exists
                    df_temp.loc[first_valid_index, 'TR'] = df_temp.loc[first_valid_index, 'H-L'] # First TR is H-L
            atr_series = df_temp['TR'].ewm(alpha=1 / period, adjust=False, min_periods=max(1, period)).mean()
        except Exception as e_pd_atr: # pragma: no cover
            atr_logger.error(f"   (Error) Pandas EWM ATR calculation failed: {e_pd_atr}", exc_info=True)
            df_result[atr_col_name] = np.nan
            df_result[atr_shifted_col_name] = np.nan
            df_result[atr_col_name] = df_result[atr_col_name].astype('float32')
            df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32')
            del df_temp
            gc.collect()
            return df_result

    df_result[atr_col_name] = atr_series.reindex(df_in.index).astype('float32')
    df_result[atr_shifted_col_name] = atr_series.shift(1).reindex(df_in.index).astype('float32')
    del df_temp, atr_series
    gc.collect()
    return df_result

def macd(series: pd.Series, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD line, signal line, and histogram."""
    macd_logger = logging.getLogger(f"{__name__}.macd")
    if not isinstance(series, pd.Series):
        macd_logger.error(f"Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        macd_logger.debug("Input series is empty, returning empty series for MACD components.")
        nan_series = pd.Series(dtype='float32')
        return nan_series, nan_series.copy(), nan_series.copy()

    nan_series_indexed = pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()

    if series_numeric.empty or len(series_numeric) < window_slow: # pragma: no cover
        macd_logger.warning(f"   (Warning) MACD calculation skipped: Not enough valid data points ({len(series_numeric)} < {window_slow}).")
        return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()

    if 'ta' not in globals() or ta is None: # pragma: no cover
        macd_logger.error("   (Error) MACD calculation failed: 'ta' library not loaded.")
        return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    try:
        macd_indicator = ta.trend.MACD(close=series_numeric, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=False)  # type: ignore
        macd_line_final = macd_indicator.macd().reindex(series.index).ffill().astype('float32')
        macd_signal_final = macd_indicator.macd_signal().reindex(series.index).ffill().astype('float32')
        macd_diff_final = macd_indicator.macd_diff().reindex(series.index).ffill().astype('float32')
        del series_numeric, macd_indicator
        gc.collect()
        return (macd_line_final, macd_signal_final, macd_diff_final)
    except Exception as e: # pragma: no cover
        macd_logger.error(f"   (Error) MACD calculation error: {e}.", exc_info=True)
        return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()

def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Calculates Rolling Z-Score."""
    zscore_logger = logging.getLogger(f"{__name__}.rolling_zscore")
    if not isinstance(series, pd.Series):
        zscore_logger.error(f"Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        zscore_logger.debug("Input series empty, returning empty series.")
        return pd.Series(dtype='float32')
    if len(series) < 2: # Z-score needs at least 2 points to calculate std
        zscore_logger.debug("Input series too short (< 2), returning zeros.")
        return pd.Series(0.0, index=series.index, dtype='float32')

    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    if series_numeric.isnull().all(): # pragma: no cover
        zscore_logger.warning("Series contains only NaN values after numeric conversion and fill, returning zeros.")
        return pd.Series(0.0, index=series.index, dtype='float32')

    actual_window = min(window, len(series_numeric))
    if actual_window < 2: # Ensure window is at least 2
        zscore_logger.debug(f"Adjusted window size ({actual_window}) < 2, returning zeros.")
        return pd.Series(0.0, index=series.index, dtype='float32')

    if min_periods is None: # Default min_periods, ensure it's at least 2
        min_periods = max(2, min(10, int(actual_window * 0.1)))
    else:
        min_periods = max(2, min(min_periods, actual_window)) # User-defined, but cap and ensure >= 2

    try:
        rolling_mean = series_numeric.rolling(window=actual_window, min_periods=min_periods).mean()
        rolling_std = series_numeric.rolling(window=actual_window, min_periods=min_periods).std()
        with np.errstate(divide='ignore', invalid='ignore'): # Handle division by zero in std
            rolling_std_safe = rolling_std.replace(0, np.nan) # Replace 0 std with NaN to avoid Inf
            z = (series_numeric - rolling_mean) / rolling_std_safe
        z_filled = z.fillna(0.0) # Fill NaNs from division by zero or insufficient periods with 0
        if np.isinf(z_filled).any(): # Should not happen if rolling_std_safe is used, but as a safeguard
            z_filled.replace([np.inf, -np.inf], 0.0, inplace=True) # pragma: no cover
        z_final = z_filled.reindex(series.index).fillna(0.0) # Ensure result has original index, fill any new NaNs
        del series_numeric, rolling_mean, rolling_std, rolling_std_safe, z, z_filled
        gc.collect()
        return z_final.astype('float32')
    except Exception as e: # pragma: no cover
        zscore_logger.error(f"Rolling Z-Score calculation failed for window {window}: {e}", exc_info=True)
        return pd.Series(0.0, index=series.index, dtype='float32')

def tag_price_structure_patterns(df: pd.DataFrame, config: 'StrategyConfig') -> pd.DataFrame:  # type: ignore
    """Tags price structure patterns based on various indicators, using config."""
    pattern_logger = logging.getLogger(f"{__name__}.tag_price_structure_patterns")
    pattern_logger.info("   (Processing) Tagging price structure patterns (using StrategyConfig)...")
    if not isinstance(df, pd.DataFrame):
        pattern_logger.error("Input must be a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty: # pragma: no cover
        pattern_logger.warning("Input DataFrame is empty. Adding 'Normal' Pattern_Label.")
        df_res = df.copy()
        df_res["Pattern_Label"] = "Normal"
        df_res["Pattern_Label"] = df_res["Pattern_Label"].astype('category')
        return df_res

    # Access pattern thresholds from config
    pattern_breakout_z_thresh = config.pattern_breakout_z_thresh
    pattern_reversal_body_ratio = config.pattern_reversal_body_ratio
    pattern_strong_trend_z_thresh = config.pattern_strong_trend_z_thresh
    pattern_choppy_candle_ratio = config.pattern_choppy_candle_ratio
    pattern_choppy_wick_ratio = config.pattern_choppy_wick_ratio

    required_cols = ["Gain_Z", "High", "Low", "Close", "Open", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]
    df_patterns = df.copy()
    missing_cols_pattern = [col for col in required_cols if col not in df_patterns.columns]
    if missing_cols_pattern: # pragma: no cover
        pattern_logger.warning(f"      (Warning) Missing columns for Pattern Labeling: {missing_cols_pattern}. Setting all to 'Normal'.")
        df_patterns["Pattern_Label"] = "Normal"
        df_patterns["Pattern_Label"] = df_patterns["Pattern_Label"].astype('category')
        return df_patterns

    # Ensure numeric types and handle NaNs for calculation columns
    for col in ["Gain_Z", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]:
        df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce').fillna(0)
    for col in ["High", "Low", "Close", "Open"]: # Price columns
        df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce')
        if df_patterns[col].isnull().any(): # pragma: no cover
            df_patterns[col] = df_patterns[col].ffill().bfill() # Fill with last/next valid price if any NaN

    df_patterns["Pattern_Label"] = "Normal" # Default label
    prev_high = df_patterns["High"].shift(1)
    prev_low = df_patterns["Low"].shift(1)
    prev_gain = df_patterns["Gain"].shift(1).fillna(0)
    prev_body = df_patterns["Candle_Body"].shift(1).fillna(0)
    prev_macd_hist = df_patterns["MACD_hist"].shift(1).fillna(0)

    # Pattern conditions (using .fillna(False) to handle potential NaNs from shift)
    breakout_cond = ((df_patterns["Gain_Z"].abs() > pattern_breakout_z_thresh) | \
                     ((df_patterns["High"] > prev_high) & (df_patterns["Close"] > prev_high)) | \
                     ((df_patterns["Low"] < prev_low) & (df_patterns["Close"] < prev_low))).fillna(False)

    reversal_cond = (((prev_gain < 0) & (df_patterns["Gain"] > 0) & (df_patterns["Candle_Body"] > (prev_body * pattern_reversal_body_ratio))) | \
                     ((prev_gain > 0) & (df_patterns["Gain"] < 0) & (df_patterns["Candle_Body"] > (prev_body * pattern_reversal_body_ratio)))).fillna(False)

    inside_bar_cond = ((df_patterns["High"] < prev_high) & (df_patterns["Low"] > prev_low)).fillna(False)

    strong_trend_cond = (((df_patterns["Gain_Z"] > pattern_strong_trend_z_thresh) & (df_patterns["MACD_hist"] > 0) & (df_patterns["MACD_hist"] > prev_macd_hist)) | \
                         ((df_patterns["Gain_Z"] < -pattern_strong_trend_z_thresh) & (df_patterns["MACD_hist"] < 0) & (df_patterns["MACD_hist"] < prev_macd_hist))).fillna(False)

    choppy_cond = ((df_patterns["Candle_Ratio"] < pattern_choppy_candle_ratio) & \
                   (df_patterns["Wick_Ratio"] > pattern_choppy_wick_ratio)).fillna(False)

    # Apply labels based on priority (Breakout first, then others if still "Normal")
    df_patterns.loc[breakout_cond, "Pattern_Label"] = "Breakout"
    df_patterns.loc[reversal_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Reversal"
    df_patterns.loc[inside_bar_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "InsideBar"
    df_patterns.loc[strong_trend_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "StrongTrend"
    df_patterns.loc[choppy_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Choppy"

    pattern_logger.info(f"      Pattern Label Distribution:\n{df_patterns['Pattern_Label'].value_counts(normalize=True).round(3).to_string()}")
    df_patterns["Pattern_Label"] = df_patterns["Pattern_Label"].astype('category')
    del prev_high, prev_low, prev_gain, prev_body, prev_macd_hist
    del breakout_cond, reversal_cond, inside_bar_cond, strong_trend_cond, choppy_cond
    gc.collect()
    return df_patterns

def calculate_m15_trend_zone(df_m15: pd.DataFrame, config: 'StrategyConfig') -> pd.DataFrame:  # type: ignore
    """Calculates M15 Trend Zone (UP, DOWN, NEUTRAL) using config."""
    m15_trend_logger = logging.getLogger(f"{__name__}.calculate_m15_trend_zone")
    m15_trend_logger.info("(Processing) กำลังคำนวณ M15 Trend Zone (using StrategyConfig)...")
    if not isinstance(df_m15, pd.DataFrame):
        m15_trend_logger.error("Input must be a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")

    result_df = pd.DataFrame(index=df_m15.index)
    # <<< MODIFIED: [Patch] Ensured DataFrame boolean checks use .empty >>>
    if df_m15.empty or "Close" not in df_m15.columns: # pragma: no cover
        m15_trend_logger.warning("Input DataFrame is empty or missing 'Close' column. Returning NEUTRAL Trend_Zone.")
        result_df["Trend_Zone"] = "NEUTRAL"
        result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        return result_df

    # Access parameters from config
    m15_trend_ema_fast = config.m15_trend_ema_fast
    m15_trend_ema_slow = config.m15_trend_ema_slow
    m15_trend_rsi_period = config.m15_trend_rsi_period
    m15_trend_rsi_up = config.m15_trend_rsi_up
    m15_trend_rsi_down = config.m15_trend_rsi_down

    df = df_m15.copy()
    try:
        df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
        if df["Close"].isnull().all(): # pragma: no cover
            m15_trend_logger.warning("'Close' column is all NaN. Returning NEUTRAL Trend_Zone.")
            result_df["Trend_Zone"] = "NEUTRAL"
            result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
            return result_df

        df["EMA_Fast"] = ema(df["Close"], m15_trend_ema_fast)
        df["EMA_Slow"] = ema(df["Close"], m15_trend_ema_slow)
        df["RSI"] = rsi(df["Close"], m15_trend_rsi_period)

        df.dropna(subset=["EMA_Fast", "EMA_Slow", "RSI"], inplace=True)
        # <<< MODIFIED: [Patch] Ensured DataFrame boolean checks use .empty >>>
        if df.empty: # pragma: no cover
            m15_trend_logger.warning("DataFrame became empty after dropping NaNs from indicator calculations. Returning NEUTRAL Trend_Zone.")
            result_df["Trend_Zone"] = "NEUTRAL"
            result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
            return result_df

        is_up = (df["EMA_Fast"] > df["EMA_Slow"]) & (df["RSI"] > m15_trend_rsi_up)
        is_down = (df["EMA_Fast"] < df["EMA_Slow"]) & (df["RSI"] < m15_trend_rsi_down)

        df["Trend_Zone"] = "NEUTRAL"
        df.loc[is_up, "Trend_Zone"] = "UP"
        df.loc[is_down, "Trend_Zone"] = "DOWN"

        # <<< MODIFIED: [Patch] Ensured DataFrame boolean checks use .empty >>>
        if not df.empty: # pragma: no cover
            m15_trend_logger.info(f"   การกระจาย M15 Trend Zone:\n{df['Trend_Zone'].value_counts(normalize=True).round(3).to_string()}")

        result_df = df[["Trend_Zone"]].reindex(df_m15.index).fillna("NEUTRAL")
        result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        del df, is_up, is_down
        gc.collect()
        return result_df
    except Exception as e: # pragma: no cover
        m15_trend_logger.error(f"(Error) การคำนวณ M15 Trend Zone ล้มเหลว: {e}", exc_info=True)
        result_df_error = pd.DataFrame(index=df_m15.index)
        result_df_error["Trend_Zone"] = "NEUTRAL"
        result_df_error["Trend_Zone"] = result_df_error["Trend_Zone"].astype('category')
        return result_df_error

def get_session_tag(timestamp: pd.Timestamp, session_times_utc_config: dict) -> str:
    """Determines the trading session (Asia, London, NY, Other) for a given UTC timestamp."""
    session_logger = logging.getLogger(f"{__name__}.get_session_tag")
    if pd.isna(timestamp): # pragma: no cover
        return "N/A"
    try:
        if not isinstance(timestamp, pd.Timestamp): # pragma: no cover
            timestamp_converted = pd.to_datetime(timestamp, errors='coerce')
            if pd.isna(timestamp_converted):
                session_logger.warning(f"Invalid timestamp for session tagging: {timestamp}")
                return "Error_Tagging"
            timestamp = timestamp_converted

        # Ensure timestamp is UTC
        ts_utc = timestamp.tz_convert('UTC') if timestamp.tzinfo else timestamp.tz_localize('UTC')
        if pd.isna(ts_utc): # pragma: no cover
            session_logger.warning(f"Timestamp became NaT after UTC conversion: {timestamp}")
            return "Error_Tagging"

        hour = ts_utc.hour
        sessions = []
        for name, (start, end) in session_times_utc_config.items():
            if start <= end: # Session does not cross midnight
                if start <= hour < end:
                    sessions.append(name)
            else: # Session crosses midnight (e.g., NY close to Asia open for some brokers)
                if hour >= start or hour < end:
                    sessions.append(name)
        return "/".join(sorted(sessions)) if sessions else "Other"
    except Exception as e: # pragma: no cover
        session_logger.error(f"   (Error) Error in get_session_tag for {timestamp}: {e}", exc_info=True)
        return "Error_Tagging"

def engineer_m1_features(df_m1: pd.DataFrame, config: 'StrategyConfig', lag_features_config: dict | None = None) -> pd.DataFrame:  # type: ignore
    """Engineers features for M1 data using parameters from StrategyConfig."""
    eng_m1_logger = logging.getLogger(f"{__name__}.engineer_m1_features")
    eng_m1_logger.info("(Processing) กำลังสร้าง Features M1 (using StrategyConfig)...")
    if not isinstance(df_m1, pd.DataFrame):
        eng_m1_logger.error("Input must be a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: # pragma: no cover
        eng_m1_logger.warning("   (Warning) ข้ามการสร้าง Features M1: DataFrame ว่างเปล่า.")
        return df_m1.copy()

    # Access parameters from config
    rolling_z_window_m1 = config.rolling_z_window_m1
    atr_rolling_avg_period = config.atr_rolling_avg_period
    timeframe_minutes_m1 = config.timeframe_minutes_m1

    df = df_m1.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    if any(col not in df.columns for col in price_cols): # pragma: no cover
        eng_m1_logger.warning(f"   (Warning) ขาดคอลัมน์ราคา M1. บาง Features อาจเป็น NaN.")
        # Define base features that might be missing if price columns are absent
        base_feature_cols = ["Candle_Body", "Candle_Range", "Gain", "Candle_Ratio", "Upper_Wick", "Lower_Wick", "Wick_Length", "Wick_Ratio", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", 'Volatility_Index', 'ADX', 'RSI', 'cluster', 'spike_score', 'session', 'Pattern_Label', 'model_tag']
        for col in base_feature_cols:
            if col not in df.columns:
                df[col] = np.nan if 'Label' not in col and 'session' not in col and 'tag' not in col else "N/A"
    else:
        # Standard feature calculation
        for col in price_cols: # Ensure numeric, handle inf/nan
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=price_cols, inplace=True) # Drop rows where essential prices are NaN
        if df.empty: # pragma: no cover
            eng_m1_logger.warning("   (Warning) M1 DataFrame ว่างเปล่าหลังลบราคา NaN.")
            return df.reindex(df_m1.index) # Return empty df with original index

        df["Candle_Body"] = abs(df["Close"] - df["Open"]).astype('float32')
        df["Candle_Range"] = (df["High"] - df["Low"]).astype('float32')
        df["Gain"] = (df["Close"] - df["Open"]).astype('float32')
        df["Candle_Ratio"] = np.where(df["Candle_Range"].abs() > 1e-9, df["Candle_Body"] / df["Candle_Range"], 0.0).astype('float32')
        df["Upper_Wick"] = (df["High"] - np.maximum(df["Open"], df["Close"])).astype('float32')
        df["Lower_Wick"] = (np.minimum(df["Open"], df["Close"]) - df["Low"]).astype('float32')
        df["Wick_Length"] = (df["Upper_Wick"] + df["Lower_Wick"]).astype('float32')
        df["Wick_Ratio"] = np.where(df["Candle_Range"].abs() > 1e-9, df["Wick_Length"] / df["Candle_Range"], 0.0).astype('float32')
        df["Gain_Z"] = rolling_zscore(df["Gain"].fillna(0), window=rolling_z_window_m1)
        df["MACD_line"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
        if "MACD_hist" in df.columns and df["MACD_hist"].notna().any():
            df["MACD_hist_smooth"] = df["MACD_hist"].rolling(window=5, min_periods=1).mean().fillna(0).astype('float32')
        else: # pragma: no cover
            df["MACD_hist_smooth"] = np.nan
            eng_m1_logger.warning("      (Warning) ไม่สามารถคำนวณ MACD_hist_smooth.")
        df = atr(df, 14) # ATR_14 and ATR_14_Shifted are added here
        if "ATR_14" in df.columns and df["ATR_14"].notna().any():
            df["ATR_14_Rolling_Avg"] = sma(df["ATR_14"], atr_rolling_avg_period)
        else: # pragma: no cover
            df["ATR_14_Rolling_Avg"] = np.nan
            eng_m1_logger.warning("      (Warning) ไม่สามารถคำนวณ ATR_14_Rolling_Avg.")
        df["Candle_Speed"] = (df["Gain"] / max(timeframe_minutes_m1, 1e-6)).astype('float32') # Avoid division by zero
        df["RSI"] = rsi(df["Close"], period=14)

    # Lagged features (using resolved config)
    actual_lag_config = lag_features_config if lag_features_config is not None else config.lag_features_config
    if actual_lag_config and isinstance(actual_lag_config, dict): # pragma: no cover
        eng_m1_logger.info(f"   Applying Lag Features based on config: {actual_lag_config}")
        for feature_name_lag in actual_lag_config.get('features', []):
            if feature_name_lag in df.columns and pd.api.types.is_numeric_dtype(df[feature_name_lag]):
                for lag_val_item in actual_lag_config.get('lags', []):
                    if isinstance(lag_val_item, int) and lag_val_item > 0:
                        df[f"{feature_name_lag}_lag{lag_val_item}"] = df[feature_name_lag].shift(lag_val_item).astype('float32')
            else:
                eng_m1_logger.warning(f"      Cannot create lag for '{feature_name_lag}': not found or not numeric.")

    # Volatility Index
    if 'ATR_14' in df.columns and 'ATR_14_Rolling_Avg' in df.columns and df['ATR_14_Rolling_Avg'].notna().any():
        df['Volatility_Index'] = np.where(df['ATR_14_Rolling_Avg'].abs() > 1e-9, df['ATR_14'] / df['ATR_14_Rolling_Avg'], np.nan)
        df['Volatility_Index'] = df['Volatility_Index'].ffill().fillna(1.0).astype('float32') # Fill NaNs from division or initial period
    else: # pragma: no cover
        df['Volatility_Index'] = 1.0 # Default if ATR components are missing
        eng_m1_logger.warning("         (Warning) ไม่สามารถคำนวณ Volatility_Index. Setting to 1.0.")

    # ADX
    if all(c in df.columns for c in ['High', 'Low', 'Close']) and ta:
        try:
            if len(df.dropna(subset=['High', 'Low', 'Close'])) >= 14 * 2 + 10: # Ensure enough data for ADX
                adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=False)  # type: ignore
                df['ADX'] = adx_indicator.adx().ffill().fillna(25.0).astype('float32') # Fill initial NaNs
            else: # pragma: no cover
                df['ADX'] = 25.0 # Default if not enough data
                eng_m1_logger.warning("         (Warning) ไม่สามารถคำนวณ ADX: ข้อมูลไม่เพียงพอ. Setting to 25.0.")
        except Exception as e_adx: # pragma: no cover
            df['ADX'] = 25.0 # Default on error
            eng_m1_logger.warning(f"         (Warning) ไม่สามารถคำนวณ ADX: {e_adx}. Setting to 25.0.")
    else: # pragma: no cover
        df['ADX'] = 25.0 # Default if HLC or ta missing

    # Price Structure Patterns (uses config for thresholds)
    df = tag_price_structure_patterns(df, config) # Pass config here

    # Contextual Clustering (KMeans) - ensure this doesn't fail with too few samples
    if 'cluster' not in df.columns: # pragma: no cover
        try:
            cluster_features = ['Gain_Z', 'Volatility_Index', 'Candle_Ratio', 'RSI', 'ADX']
            features_present = [f_cluster for f_cluster in cluster_features if f_cluster in df.columns and df[f_cluster].notna().any()]
            if len(features_present) < 2 or len(df[features_present].dropna()) < 3: # Need at least 2 features and 3 samples
                df['cluster'] = 0
                eng_m1_logger.warning("         (Warning) Not enough valid features/samples for clustering. Setting cluster to 0.")
            else:
                X_cluster_raw = df[features_present].copy().replace([np.inf, -np.inf], np.nan)
                X_cluster = X_cluster_raw.fillna(X_cluster_raw.median()).fillna(0) # Fill NaNs robustly
                if len(X_cluster) >= 3: # Check again after fillna
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_cluster)
                    kmeans = KMeans(n_clusters=min(3, len(X_cluster)), random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10)
                    df['cluster'] = kmeans.fit_predict(X_scaled)
                else:
                    df['cluster'] = 0
                    eng_m1_logger.warning("         (Warning) Not enough samples after cleaning for clustering. Setting cluster to 0.")
        except Exception as e_cluster:
            df['cluster'] = 0 # Default on error
            eng_m1_logger.error(f"         (Error) Clustering failed: {e_cluster}.", exc_info=True)
        if 'cluster' in df.columns:
            df['cluster'] = pd.to_numeric(df['cluster'], downcast='integer')

    # Spike Score (Simple heuristic)
    if 'spike_score' not in df.columns: # pragma: no cover
        try:
            gain_z_abs_spike = abs(pd.to_numeric(df.get('Gain_Z', 0.0), errors='coerce').fillna(0.0))
            wick_ratio_val_spike = abs(pd.to_numeric(df.get('Wick_Ratio', 0.0), errors='coerce').fillna(0.0))
            atr_val_for_spike = pd.to_numeric(df.get('ATR_14', 1.0), errors='coerce').fillna(1.0).replace([np.inf, -np.inf], 1.0) # Ensure ATR is positive
            score_spike = (wick_ratio_val_spike * 0.5 + gain_z_abs_spike * 0.3 + atr_val_for_spike * 0.2)
            # Boost score if ATR is high and wick ratio is high (indicative of strong spike)
            score_spike = np.where((atr_val_for_spike > 1.5) & (wick_ratio_val_spike > 0.6), score_spike * 1.2, score_spike)
            df['spike_score'] = score_spike.clip(0, 1).astype('float32')
        except Exception as e_spike:
            df['spike_score'] = 0.0 # Default on error
            eng_m1_logger.error(f"         (Error) Spike score calculation failed: {e_spike}.", exc_info=True)

    # Session Tag
    if 'session' not in df.columns: # pragma: no cover
        eng_m1_logger.info("      Creating 'session' column using config.session_times_utc...")
        try:
            original_index_is_datetime = isinstance(df.index, pd.DatetimeIndex) and not df.index.hasnans
            if original_index_is_datetime:
                if not df.empty: # Check if DataFrame is not empty before applying
                    df['session'] = df.index.to_series().apply(lambda ts: get_session_tag(ts, config.session_times_utc))
                else:
                    df['session'] = "N/A_EmptyDF" # Handle empty DataFrame case
            else:
                eng_m1_logger.warning("         (Warning) Original index is not a valid DatetimeIndex. Session tagging will result in 'Error_Tagging_Reindex_Fill'.")
                temp_index_for_apply_session = pd.to_datetime(df.index, errors='coerce')
                if not temp_index_for_apply_session.isna().all():
                    session_values_on_temp_index_session = temp_index_for_apply_session.to_series().apply(lambda ts: get_session_tag(ts, config.session_times_utc))
                    df['session'] = session_values_on_temp_index_session.reindex(df.index) # Reindex back
                    df['session'] = df['session'].fillna("Error_Tagging_Reindex_Fill") # Fill any new NaNs
                else:
                    df['session'] = "Error_Index_Conv"
            df['session'] = df['session'].astype('category')
            if not df.empty: # Check again before value_counts
                eng_m1_logger.info(f"         Session distribution:\n{df['session'].value_counts(normalize=True).round(3).to_string()}")
        except Exception as e_session:
            eng_m1_logger.error(f"         (Error) Session calculation failed: {e_session}. Assigning 'Other'.", exc_info=True)
            df['session'] = "Other"
            df['session'] = df['session'].astype('category')

    # Ensure 'model_tag' exists, default to 'N/A'
    if 'model_tag' not in df.columns: # pragma: no cover
        df['model_tag'] = 'N/A'

    eng_m1_logger.info("(Success) สร้าง Features M1 (using StrategyConfig) เสร็จสิ้น.")
    return df.reindex(df_m1.index) # Reindex to original to ensure no rows are lost/gained if NaNs were dropped internally

def clean_m1_data(df_m1: pd.DataFrame, config: 'StrategyConfig') -> tuple[pd.DataFrame, list]:  # type: ignore
    """Cleans M1 data, converts types, and identifies features for drift analysis, using config."""
    clean_logger = logging.getLogger(f"{__name__}.clean_m1_data")
    clean_logger.info("(Processing) กำหนด Features M1 สำหรับ Drift และแปลงประเภท (using StrategyConfig)...")
    if not isinstance(df_m1, pd.DataFrame):
        clean_logger.error("Input must be a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: # pragma: no cover
        clean_logger.warning("   (Warning) ข้ามการทำความสะอาดข้อมูล M1: DataFrame ว่างเปล่า.")
        return pd.DataFrame(), []

    df_cleaned = df_m1.copy()
    meta_features_from_config = config.meta_classifier_features # Get from config

    # Define a base list of potential features, add any from config if not present
    potential_m1_features_list = [
        "Candle_Body", "Candle_Range", "Candle_Ratio", "Gain", "Gain_Z",
        "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth",
        "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed",
        "Wick_Length", "Wick_Ratio", "Pattern_Label", "Signal_Score",
        'Volatility_Index', 'ADX', 'RSI', 'cluster', 'spike_score', 'session'
        # 'model_tag' is usually for output, not drift input feature
    ]
    # Add any lag features that might have been created
    lag_cols_in_df_clean = [col for col in df_cleaned.columns if '_lag' in col]
    potential_m1_features_list.extend(lag_cols_in_df_clean)
    if meta_features_from_config: # Ensure meta_features are considered if defined
        potential_m1_features_list.extend([f for f in meta_features_from_config if f not in potential_m1_features_list])
    potential_m1_features_list = sorted(list(dict.fromkeys(potential_m1_features_list))) # Unique and sorted
    m1_features_for_drift_local_clean = [f for f in potential_m1_features_list if f in df_cleaned.columns]
    clean_logger.info(f"   กำหนด {len(m1_features_for_drift_local_clean)} Features M1 สำหรับ Drift: {m1_features_for_drift_local_clean}")

    # Type conversion and NaN/Inf handling for numeric columns
    numeric_cols_clean = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_clean:
        try:
            # Replace Inf with NaN first
            inf_mask_clean = df_cleaned[numeric_cols_clean].isin([np.inf, -np.inf])
            if inf_mask_clean.any().any(): # pragma: no cover
                cols_with_inf_clean = df_cleaned[numeric_cols_clean].columns[inf_mask_clean.any()].tolist()
                clean_logger.warning(f"      [Inf Check] พบ Inf ใน: {cols_with_inf_clean}. กำลังแทนที่ด้วย NaN...")
                df_cleaned[cols_with_inf_clean] = df_cleaned[cols_with_inf_clean].replace([np.inf, -np.inf], np.nan)

            # Ensure all intended numeric columns are actually numeric (some might be object)
            for col_clean in numeric_cols_clean: # Ensure all numeric cols are actually numeric
                df_cleaned[col_clean] = pd.to_numeric(df_cleaned[col_clean], errors='coerce')

            # Fill NaNs in numeric columns
            cols_with_nan_clean = df_cleaned[numeric_cols_clean].columns[df_cleaned[numeric_cols_clean].isnull().any()].tolist()
            if cols_with_nan_clean: # pragma: no cover
                clean_logger.info(f"      [NaN Check] พบ NaN ใน: {cols_with_nan_clean}. กำลังเติมด้วย ffill().fillna(0)...")
                df_cleaned[cols_with_nan_clean] = df_cleaned[cols_with_nan_clean].ffill().fillna(0) # Robust fill

            # Optimize numeric types
            for col_clean_type in numeric_cols_clean:
                if col_clean_type not in df_cleaned.columns: # Column might have been dropped if all NaN
                    continue
                if pd.api.types.is_integer_dtype(df_cleaned[col_clean_type].dtype):
                    df_cleaned[col_clean_type] = pd.to_numeric(df_cleaned[col_clean_type], downcast='integer')
                elif pd.api.types.is_float_dtype(df_cleaned[col_clean_type].dtype) and df_cleaned[col_clean_type].dtype != 'float32':
                    df_cleaned[col_clean_type] = df_cleaned[col_clean_type].astype('float32')
        except Exception as e_clean_numeric: # pragma: no cover
            clean_logger.error(f"   (Error) เกิดข้อผิดพลาดในการแปลงประเภทข้อมูลหรือเติม NaN/Inf: {e_clean_numeric}.", exc_info=True)

    # Convert known categorical columns
    categorical_cols_clean = ['Pattern_Label', 'session'] # Add other known categoricals if any
    for col_cat_clean in categorical_cols_clean:
        if col_cat_clean in df_cleaned.columns:
            if df_cleaned[col_cat_clean].isnull().any(): # pragma: no cover
                df_cleaned[col_cat_clean] = df_cleaned[col_cat_clean].fillna("Unknown") # Fill NaNs before astype
            if not isinstance(df_cleaned[col_cat_clean].dtype, pd.CategoricalDtype): # Avoid re-casting if already category
                try:
                    df_cleaned[col_cat_clean] = df_cleaned[col_cat_clean].astype('category')
                except Exception as e_cat_clean: # pragma: no cover
                    clean_logger.warning(f"   (Warning) เกิดข้อผิดพลาดในการแปลง '{col_cat_clean}' เป็น category: {e_cat_clean}.")

    clean_logger.info("(Success) กำหนด Features M1 และแปลงประเภท (using StrategyConfig) เสร็จสิ้น.")
    return df_cleaned, m1_features_for_drift_local_clean

def calculate_m1_entry_signals(df_m1: pd.DataFrame, fold_specific_config: dict, strategy_config: 'StrategyConfig') -> pd.DataFrame:  # type: ignore
    """Calculates M1 entry signals based on fold-specific and global strategy configurations."""
    signal_logger = logging.getLogger(f"{__name__}.calculate_m1_entry_signals")
    signal_logger.debug("      (Calculating M1 Signals using StrategyConfig)...")
    df = df_m1.copy()
    df['Signal_Score'] = 0.0  # Initialize

    # Parameters from fold_specific_config (overrides) or defaults from strategy_config
    gain_z_thresh_signal = fold_specific_config.get('gain_z_thresh', strategy_config.default_gain_z_thresh_fold)
    rsi_thresh_buy_signal = fold_specific_config.get('rsi_thresh_buy', strategy_config.default_rsi_thresh_buy_fold)
    rsi_thresh_sell_signal = fold_specific_config.get('rsi_thresh_sell', strategy_config.default_rsi_thresh_sell_fold)
    volatility_max_signal = fold_specific_config.get('volatility_max', strategy_config.default_volatility_max_fold)
    ignore_rsi_scoring_signal = fold_specific_config.get('ignore_rsi_scoring', strategy_config.default_ignore_rsi_scoring_fold)

    entry_score_min_signal = strategy_config.min_signal_score_entry # From global config

    # Ensure required columns exist and are numeric, fillna if necessary
    # Using .get() with a default Series handles missing columns gracefully.
    df['Gain_Z'] = df.get('Gain_Z', pd.Series(0.0, index=df.index)).fillna(0.0)
    buy_gain_z_cond_signal = df['Gain_Z'] > gain_z_thresh_signal
    sell_gain_z_cond_signal = df['Gain_Z'] < -gain_z_thresh_signal

    df['Pattern_Label'] = df.get('Pattern_Label', pd.Series('Normal', index=df.index)).astype(str).fillna('Normal')
    # Example pattern logic - this can be more complex
    buy_pattern_cond_signal = df['Pattern_Label'].isin(['Breakout', 'StrongTrend']) & (df['Gain_Z'] > 0) # Example logic
    sell_pattern_cond_signal = df['Pattern_Label'].isin(['Breakout', 'StrongTrend', 'Reversal']) & (df['Gain_Z'] < 0) # Example logic

    df['RSI'] = df.get('RSI', pd.Series(50.0, index=df.index)).fillna(50.0)
    buy_rsi_cond_signal = df['RSI'] > rsi_thresh_buy_signal
    sell_rsi_cond_signal = df['RSI'] < rsi_thresh_sell_signal

    df['Volatility_Index'] = df.get('Volatility_Index', pd.Series(1.0, index=df.index)).fillna(1.0)
    vol_cond_signal = df['Volatility_Index'] < volatility_max_signal # Example: Lower volatility is better for some strategies

    # Scoring logic (example)
    df.loc[buy_gain_z_cond_signal, 'Signal_Score'] += 1.0
    df.loc[sell_gain_z_cond_signal, 'Signal_Score'] -= 1.0
    df.loc[buy_pattern_cond_signal, 'Signal_Score'] += 1.0
    df.loc[sell_pattern_cond_signal, 'Signal_Score'] -= 1.0
    if not ignore_rsi_scoring_signal:
        df.loc[buy_rsi_cond_signal, 'Signal_Score'] += 1.0
        df.loc[sell_rsi_cond_signal, 'Signal_Score'] -= 1.0
    df.loc[vol_cond_signal, 'Signal_Score'] += 1.0 # Add score if volatility is favorable

    df['Signal_Score'] = df['Signal_Score'].astype('float32')
    df['Entry_Long'] = ((df['Signal_Score'] > 0) & (df['Signal_Score'] >= entry_score_min_signal)).astype(int)
    df['Entry_Short'] = ((df['Signal_Score'] < 0) & (abs(df['Signal_Score']) >= entry_score_min_signal)).astype(int)

    # Create Trade_Reason string
    df['Trade_Reason'] = ""
    df.loc[buy_gain_z_cond_signal, 'Trade_Reason'] += f"+Gz>{gain_z_thresh_signal:.1f}"
    df.loc[sell_gain_z_cond_signal, 'Trade_Reason'] += f"+Gz<{-gain_z_thresh_signal:.1f}"
    df.loc[buy_pattern_cond_signal, 'Trade_Reason'] += "+PBuy"
    df.loc[sell_pattern_cond_signal, 'Trade_Reason'] += "+PSell"
    if not ignore_rsi_scoring_signal:
        df.loc[buy_rsi_cond_signal, 'Trade_Reason'] += f"+RSI>{rsi_thresh_buy_signal}"
        df.loc[sell_rsi_cond_signal, 'Trade_Reason'] += f"+RSI<{rsi_thresh_sell_signal}"
    df.loc[vol_cond_signal, 'Trade_Reason'] += f"+Vol<{volatility_max_signal:.1f}"

    buy_entry_mask_signal = df['Entry_Long'] == 1
    sell_entry_mask_signal = df['Entry_Short'] == 1
    df.loc[buy_entry_mask_signal, 'Trade_Reason'] = "BUY(" + df.loc[buy_entry_mask_signal, 'Signal_Score'].round(1).astype(str) + "):" + df.loc[buy_entry_mask_signal, 'Trade_Reason'].str.lstrip('+')
    df.loc[sell_entry_mask_signal, 'Trade_Reason'] = "SELL(" + df.loc[sell_entry_mask_signal, 'Signal_Score'].abs().round(1).astype(str) + "):" + df.loc[sell_entry_mask_signal, 'Trade_Reason'].str.lstrip('+')
    df.loc[~(buy_entry_mask_signal | sell_entry_mask_signal), 'Trade_Reason'] = "NONE"

    df['Trade_Tag'] = df['Signal_Score'].round(1).astype(str) + "_" + df['Pattern_Label'].astype(str)
    return df

logger.info("Part 6 (Original Part 5): Feature Engineering & Indicator Calculation Functions Loaded and Refactored to use StrategyConfig.")
# === END OF PART 6/15 ===
# === START OF PART 7/15 ===
# ==============================================================================
# === PART 7: Machine Learning Configuration & Helpers (v4.9.13 - Refined Model Switcher Fallback) ===
# ==============================================================================
# <<< MODIFIED: select_model_for_trade now returns (None, None) if no valid model (including 'main') can be selected. >>>

import logging  # Already imported
import os  # Already imported
import json  # Already imported
import pandas as pd  # Already imported
import numpy as np  # Already imported
import matplotlib.pyplot as plt  # type: ignore # Already imported
import traceback  # Already imported

# Import ML libraries conditionally (assuming they are checked/installed in Part 1)
try:
    import shap
except ImportError:
    shap = None  # type: ignore
try:
    from catboost import CatBoostClassifier, Pool  # type: ignore
except ImportError:
    CatBoostClassifier = None  # type: ignore
    Pool = None  # type: ignore
# sklearn.metrics are imported in Part 1

# --- ML Model Usage Flags & Paths (Defaults, can be overridden by StrategyConfig or main flow) ---
USE_META_CLASSIFIER = True
USE_META_META_CLASSIFIER = False # Not actively used, placeholder
ENABLE_OPTUNA_TUNING = False # Default, can be overridden by config
ENABLE_AUTO_THRESHOLD_TUNING = False # Not actively used, placeholder

# --- Global variables to store model info (populated at runtime) ---
meta_model_type_used: str = "N/A" # Stores the type of L1 model used (e.g., CatBoost)
meta_meta_model_type_used: str = "N/A" # Stores the type of L2 model used

ml_helper_logger = logging.getLogger(f"{__name__}.MLHelpers")
ml_helper_logger.info("Loading Machine Learning Configuration & Helpers (v4.9.13 - Refined Model Switcher)...")
ml_helper_logger.info(f"  Initial USE_META_CLASSIFIER (L1 Filter): {USE_META_CLASSIFIER}")
ml_helper_logger.info(f"  Initial USE_META_META_CLASSIFIER (L2 Filter): {USE_META_META_CLASSIFIER}")
ml_helper_logger.info(f"  Initial Optuna Hyperparameter Tuning Enabled: {ENABLE_OPTUNA_TUNING}")
ml_helper_logger.info(f"  Initial Auto Threshold Tuning Enabled: {ENABLE_AUTO_THRESHOLD_TUNING}")

# --- SHAP Feature Selection Helper Function ---
def select_top_shap_features(shap_values_val: np.ndarray | list | None,
                             feature_names: list[str] | None,
                             shap_threshold: float = 0.01) -> list[str] | None:
    shap_select_logger = logging.getLogger(f"{__name__}.select_top_shap_features")
    shap_select_logger.info(f"   [SHAP Select] กำลังเลือก Features ที่มี Normalized SHAP >= {shap_threshold:.4f}...")

    if shap_values_val is None or not isinstance(shap_values_val, (np.ndarray, list)) or \
       (isinstance(shap_values_val, np.ndarray) and shap_values_val.size == 0) or \
       (isinstance(shap_values_val, list) and not shap_values_val): # pragma: no cover
        shap_select_logger.warning("      (Warning) ไม่สามารถเลือก Features: ค่า SHAP ไม่ถูกต้องหรือว่างเปล่า. คืนค่า Features เดิม.")
        return feature_names if isinstance(feature_names, list) else []

    if feature_names is None or not isinstance(feature_names, list) or not feature_names: # pragma: no cover
        shap_select_logger.warning("      (Warning) ไม่สามารถเลือก Features: รายชื่อ Features ไม่ถูกต้องหรือว่างเปล่า. คืนค่า None.")
        return None

    shap_values_to_process = None
    if isinstance(shap_values_val, list):
        if len(shap_values_val) >= 2 and isinstance(shap_values_val[1], np.ndarray): # Common for binary classification SHAP
            shap_select_logger.debug("      (Info) SHAP values appear to be for multiple classes, using index 1 (positive class).")
            shap_values_to_process = shap_values_val[1]
        elif len(shap_values_val) == 1 and isinstance(shap_values_val[0], np.ndarray): # Single output (e.g., regression or single class)
            shap_values_to_process = shap_values_val[0]
        else: # pragma: no cover
            shap_select_logger.warning(f"      (Warning) SHAP values list has unexpected structure. คืนค่า Features เดิม.")
            return feature_names
    elif isinstance(shap_values_val, np.ndarray):
        shap_values_to_process = shap_values_val
    else: # pragma: no cover
        shap_select_logger.warning(f"      (Warning) SHAP values has unexpected type: {type(shap_values_val)}. คืนค่า Features เดิม.")
        return feature_names

    if shap_values_to_process is None or shap_values_to_process.ndim != 2: # pragma: no cover
        shap_select_logger.warning(f"      (Warning) ขนาด SHAP values ที่จะประมวลผลไม่ถูกต้อง (Dimensions: {shap_values_to_process.ndim if shap_values_to_process is not None else 'N/A'}, expected 2). คืนค่า Features เดิม.")
        return feature_names
    if shap_values_to_process.shape[1] != len(feature_names): # pragma: no cover
        shap_select_logger.warning(f"      (Warning) ขนาด SHAP values ไม่ตรงกับจำนวน Features (SHAP: {shap_values_to_process.shape[1]}, Features: {len(feature_names)}). คืนค่า Features เดิม.")
        return feature_names
    if shap_values_to_process.shape[0] == 0: # pragma: no cover
        shap_select_logger.warning("      (Warning) SHAP values array มี 0 samples. ไม่สามารถคำนวณ Importance ได้. คืนค่า Features เดิม.")
        return feature_names

    try:
        mean_abs_shap = np.abs(shap_values_to_process).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any(): # pragma: no cover
            shap_select_logger.warning("      (Warning) พบ NaN หรือ Inf ใน Mean Absolute SHAP values. ไม่สามารถเลือก Features ได้. คืนค่า Features เดิม.")
            return feature_names

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        if total_shap > 1e-9: # Avoid division by zero
            shap_df["Normalized_SHAP"] = shap_df["Mean_Abs_SHAP"] / total_shap
        else:
            shap_df["Normalized_SHAP"] = 0.0 # If total SHAP is ~0, all normalized will be 0
            shap_select_logger.warning("      (Warning) Total Mean Abs SHAP ใกล้ศูนย์, ไม่สามารถ Normalize ได้. จะไม่เลือก Feature ใดๆ.")
            return [] # No features selected if total importance is zero

        selected_features_df = shap_df[shap_df["Normalized_SHAP"] >= shap_threshold].copy()
        selected_features_list = selected_features_df["Feature"].tolist()

        if not selected_features_list: # pragma: no cover
            shap_select_logger.warning(f"      (Warning) ไม่มี Features ใดผ่านเกณฑ์ SHAP >= {shap_threshold:.4f}. คืนค่า List ว่าง.")
            return []
        elif len(selected_features_list) < len(feature_names): # pragma: no cover
            removed_features = sorted(list(set(feature_names) - set(selected_features_list)))
            shap_select_logger.info(f"      (Success) เลือก Features ได้ {len(selected_features_list)} ตัวจาก SHAP.")
            shap_select_logger.info(f"         Features ที่ถูกตัดออก {len(removed_features)} ตัว: {removed_features}")
            shap_select_logger.info("         Features ที่เลือก (เรียงตามค่า SHAP):")
            shap_select_logger.info("\n" + selected_features_df.sort_values("Normalized_SHAP", ascending=False)[["Feature", "Normalized_SHAP"]].round(5).to_string(index=False))
        else:
            shap_select_logger.info("      (Success) Features ทั้งหมดผ่านเกณฑ์ SHAP.") # All features kept
        return selected_features_list
    except Exception as e: # pragma: no cover
        shap_select_logger.error(f"      (Error) เกิดข้อผิดพลาดระหว่างการเลือก Features ด้วย SHAP: {e}. คืนค่า Features เดิม.", exc_info=True)
        return feature_names

# --- Model Quality Check Functions ---
def check_model_overfit(model, X_train, y_train, X_val, y_val, metric: str = "AUC", threshold_pct: float = 10.0):
    overfit_logger = logging.getLogger(f"{__name__}.check_model_overfit")
    try:
        overfit_logger.info(f"[Check] Checking for Overfitting ({metric})...")

        if model is None: # pragma: no cover
            overfit_logger.warning("[OverfitCheck] Model is None. Cannot perform overfit check.")
            return
        if X_train is None or y_train is None or X_val is None or y_val is None: # pragma: no cover
            overfit_logger.warning("[OverfitCheck] Training or Validation data is None. Cannot perform overfit check.")
            return
        if len(X_train) == 0 or len(X_val) == 0: # pragma: no cover
            overfit_logger.warning("[OverfitCheck] Training or Validation data is empty. Cannot perform overfit check.")
            return

        train_score = np.nan
        val_score = np.nan

        if metric == "AUC":
            if not hasattr(model, 'predict_proba'): # pragma: no cover
                overfit_logger.warning(f"      (Warning) Model for AUC check does not have 'predict_proba' method.")
                return
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            train_score = roc_auc_score(y_train, y_train_proba)  # type: ignore
            val_score = roc_auc_score(y_val, y_val_proba)  # type: ignore
        elif metric == "Accuracy":
            if not hasattr(model, 'predict'): # pragma: no cover
                overfit_logger.warning(f"      (Warning) Model for Accuracy check does not have 'predict' method.")
                return
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_score = accuracy_score(y_train, y_train_pred)  # type: ignore
            val_score = accuracy_score(y_val, y_val_pred)  # type: ignore
        elif metric == "LogLoss":
            if not hasattr(model, 'predict_proba'): # pragma: no cover
                overfit_logger.warning(f"      (Warning) Model for LogLoss check does not have 'predict_proba' method.")
                return
            y_train_proba_ll = model.predict_proba(X_train)
            y_val_proba_ll = model.predict_proba(X_val)
            train_score = log_loss(y_train, y_train_proba_ll)  # type: ignore
            val_score = log_loss(y_val, y_val_proba_ll)  # type: ignore
        else: # pragma: no cover
            overfit_logger.warning(f"[OverfitCheck] Unknown metric '{metric}' used.")
            return

        if pd.isna(train_score) or pd.isna(val_score): # pragma: no cover
            overfit_logger.warning(f"[OverfitCheck] Could not calculate scores for metric '{metric}'. Train: {train_score}, Val: {val_score}")
            return

        overfit_detected = False
        if metric in ["AUC", "Accuracy"]: # Higher is better
            diff = train_score - val_score
            threshold_abs = abs(train_score) * (threshold_pct / 100.0) if abs(train_score) > 1e-9 else threshold_pct / 100.0
            if diff >= threshold_abs:
                overfit_detected = True
        elif metric == "LogLoss": # Lower is better
            diff = val_score - train_score # Positive diff means val_score is worse (higher)
            threshold_abs = abs(train_score) * (threshold_pct / 100.0) if abs(train_score) > 1e-9 else threshold_pct / 100.0
            if diff >= threshold_abs:
                overfit_detected = True
        else: # pragma: no cover
            return # Should have been caught by unknown metric check

        diff_percent = (diff / abs(train_score) * 100.0) if abs(train_score) > 1e-9 else float('inf') if diff != 0 else 0.0
        overfit_logger.info(f"[OverfitCheck] Train {metric}: {train_score:.4f}, Val {metric}: {val_score:.4f}, Diff: {diff:.4f} ({diff_percent:.1f}%), Threshold_abs: {threshold_abs:.4f}")

        if overfit_detected: # pragma: no cover
            overfit_logger.warning(f"[Patch] Potential Overfitting detected ({metric}): Train={train_score:.3f}, Val={val_score:.3f}, Diff={diff:.3f}")
    except Exception as e: # pragma: no cover
        overfit_logger.error(f"[OverfitCheck] Error during overfit check: {str(e)}", exc_info=True)

def check_feature_noise_shap(shap_values: np.ndarray | None, feature_names: list[str] | None, threshold: float = 0.01):
    noise_logger = logging.getLogger(f"{__name__}.check_feature_noise_shap")
    noise_logger.info("   [Check] Checking for Feature Noise (SHAP)...")
    if shap_values is None or not isinstance(shap_values, np.ndarray) or not feature_names or not isinstance(feature_names, list) or \
       shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names) or shap_values.shape[0] == 0: # pragma: no cover
        noise_logger.warning("      (Warning) Skipping Feature Noise Check: Invalid inputs."); return

    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any(): # pragma: no cover
            noise_logger.warning("      (Warning) Found NaN/Inf in Mean Abs SHAP. Skipping noise check.")
            return

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        shap_df["Normalized_SHAP"] = (shap_df["Mean_Abs_SHAP"] / total_shap) if total_shap > 1e-9 else 0.0

        shap_series_for_check = pd.Series(shap_df["Normalized_SHAP"].values, index=shap_df["Feature"])
        noise_feats = shap_series_for_check[shap_series_for_check < threshold].index.tolist()
        if noise_feats: # pragma: no cover
            noise_logger.info(f"[Patch] SHAP Noise features detected (Normalized SHAP < {threshold:.4f}): {noise_feats}")
        else:
            noise_logger.info(f"      (Success) No features with significant noise detected (Normalized SHAP < {threshold:.4f}).")
    except Exception as e: # pragma: no cover
        noise_logger.error(f"      (Error) Error during Feature Noise check (SHAP): {e}", exc_info=True)

# --- SHAP Analysis Function ---
def analyze_feature_importance_shap(model, model_type: str, data_sample: pd.DataFrame | None,
                                    features: list[str] | None, output_dir: str, fold_idx: int | None = None):
    shap_analyze_logger = logging.getLogger(f"{__name__}.analyze_feature_importance_shap")
    global shap # Access global shap (which might be None)
    if not shap: # pragma: no cover
        shap_analyze_logger.warning("   (Warning) Skipping SHAP: 'shap' library not found.")
        return
    if model is None: # pragma: no cover
        shap_analyze_logger.warning("   (Warning) Skipping SHAP: Model is None.")
        return
    if data_sample is None or not isinstance(data_sample, pd.DataFrame) or data_sample.empty: # pragma: no cover
        shap_analyze_logger.warning("   (Warning) Skipping SHAP: No sample data.")
        return
    if not features or not isinstance(features, list) or not all(isinstance(f, str) for f in features): # pragma: no cover
        shap_analyze_logger.warning("   (Warning) Skipping SHAP: Invalid features list.")
        return
    if not output_dir or not os.path.isdir(output_dir): # pragma: no cover
        shap_analyze_logger.warning(f"   (Warning) Skipping SHAP: Output directory '{output_dir}' invalid.")
        return

    fold_suffix = f"_fold{fold_idx+1}" if fold_idx is not None else "_validation_set"
    shap_analyze_logger.info(f"\n(Analyzing) SHAP analysis ({model_type} - Sample Size: {len(data_sample)}) - {fold_suffix.replace('_',' ').title()}...")

    missing_features_shap = [f for f in features if f not in data_sample.columns]
    if missing_features_shap: # pragma: no cover
        shap_analyze_logger.error(f"   (Error) Skipping SHAP: Missing features in data_sample: {missing_features_shap}")
        return
    try:
        X_shap = data_sample[features].copy()
    except KeyError as e_key_shap: # pragma: no cover
        shap_analyze_logger.error(f"   (Error) Skipping SHAP: Feature(s) not found: {e_key_shap}")
        return
    except Exception as e_select_shap: # pragma: no cover
        shap_analyze_logger.error(f"   (Error) Skipping SHAP: Error selecting features: {e_select_shap}", exc_info=True)
        return

    # Handle categorical features for SHAP Pool if CatBoost
    cat_features_indices_shap = []
    cat_feature_names_for_shap = []
    potential_cat_cols_shap = ['Pattern_Label', 'session', 'Trend_Zone'] # Add others if known
    shap_analyze_logger.debug("      Processing categorical features for SHAP...")
    for cat_col_shap in potential_cat_cols_shap:
        if cat_col_shap in X_shap.columns:
            try:
                if X_shap[cat_col_shap].isnull().any(): # pragma: no cover
                    X_shap[cat_col_shap].fillna("Missing_SHAP_Fill", inplace=True) # Fill NaNs in categoricals
                X_shap[cat_col_shap] = X_shap[cat_col_shap].astype(str) # Ensure string for CatBoost
                if model_type == "CatBoostClassifier":
                    cat_feature_names_for_shap.append(cat_col_shap)
            except Exception as e_cat_str_shap: # pragma: no cover
                shap_analyze_logger.warning(f"      (Warning) Could not convert '{cat_col_shap}' to string for SHAP: {e_cat_str_shap}.")

    if model_type == "CatBoostClassifier" and cat_feature_names_for_shap:
        try:
            cat_features_indices_shap = [X_shap.columns.get_loc(col) for col in cat_feature_names_for_shap]
            shap_analyze_logger.debug(f"         Categorical Feature Indices for SHAP Pool: {cat_features_indices_shap}")
        except KeyError as e_loc_shap: # pragma: no cover
            shap_analyze_logger.error(f"      (Error) Could not locate categorical feature index for SHAP: {e_loc_shap}.")
            cat_features_indices_shap = [] # Reset if error

    # Handle NaN/Inf in numeric features before SHAP
    shap_analyze_logger.debug("      Handling NaN/Inf in numeric features for SHAP...")
    numeric_cols_for_shap = X_shap.select_dtypes(include=np.number).columns
    if X_shap[numeric_cols_for_shap].isin([np.inf, -np.inf]).any().any(): # pragma: no cover
        X_shap[numeric_cols_for_shap] = X_shap[numeric_cols_for_shap].replace([np.inf, -np.inf], np.nan)
    if X_shap[numeric_cols_for_shap].isnull().any().any(): # pragma: no cover
        X_shap[numeric_cols_for_shap] = X_shap[numeric_cols_for_shap].fillna(0) # Fill with 0, or consider median/mean

    if X_shap.isnull().any().any(): # pragma: no cover
        missing_final_shap = X_shap.columns[X_shap.isnull().any()].tolist()
        shap_analyze_logger.error(f"      (Error) Skipping SHAP: NaNs still present after fill in columns: {missing_final_shap}")
        return

    try:
        explainer_shap_obj = None
        shap_values_calculated = None
        global CatBoostClassifier, Pool # Access globals that might be None

        shap_analyze_logger.debug(f"      Initializing SHAP explainer for model type: {model_type}...")
        if model_type == "CatBoostClassifier" and CatBoostClassifier and Pool:
            shap_pool_obj = Pool(X_shap, label=None, cat_features=cat_features_indices_shap) # type: ignore
            explainer_shap_obj = shap.TreeExplainer(model)  # type: ignore
            shap_analyze_logger.info(f"      Calculating SHAP values (CatBoost)...")
            shap_values_calculated = explainer_shap_obj.shap_values(shap_pool_obj)
        # Add other model types (e.g., "XGBClassifier") here if needed
        # elif model_type == "XGBClassifier" and XGBClassifier:
        # explainer_shap_obj = shap.TreeExplainer(model)
        # shap_values_calculated = explainer_shap_obj.shap_values(X_shap)
        else: # pragma: no cover
            shap_analyze_logger.warning(f"      (Warning) SHAP explainer not supported or library missing for model type: {model_type}. Skipping SHAP calculation.")
            return

        # Process SHAP values for plotting (handle different output structures)
        shap_values_positive_class_plot = None
        if isinstance(shap_values_calculated, list) and len(shap_values_calculated) >= 2: # Common for binary classification
            if isinstance(shap_values_calculated[1], np.ndarray) and shap_values_calculated[1].ndim == 2:
                shap_values_positive_class_plot = shap_values_calculated[1] # Use SHAP for positive class
            else: # pragma: no cover
                shap_analyze_logger.error(f"      (Error) SHAP values list element 1 has unexpected type/shape: {type(shap_values_calculated[1])}, {getattr(shap_values_calculated[1], 'shape', 'N/A')}")
        elif isinstance(shap_values_calculated, np.ndarray) and shap_values_calculated.ndim == 2: # Single output or already selected class
            shap_values_positive_class_plot = shap_values_calculated
        elif isinstance(shap_values_calculated, np.ndarray) and shap_values_calculated.ndim == 3: # For some model types, shape might be (n_outputs, n_samples, n_features)
            # Try to infer the positive class or primary output
            if shap_values_calculated.shape[0] >= 2 and shap_values_calculated.shape[1] == X_shap.shape[0] and shap_values_calculated.shape[2] == X_shap.shape[1]:
                shap_values_positive_class_plot = shap_values_calculated[1, :, :] # Assuming second output is positive class
            elif shap_values_calculated.shape[2] >= 2 and shap_values_calculated.shape[0] == X_shap.shape[0] and shap_values_calculated.shape[1] == X_shap.shape[1]: # (n_samples, n_features, n_outputs)
                shap_values_positive_class_plot = shap_values_calculated[:, :, 1] # Assuming third dimension, index 1 is positive class
            elif shap_values_calculated.shape[0] == 1: # Only one output class
                shap_values_positive_class_plot = shap_values_calculated[0, :, :]
                shap_analyze_logger.warning("      SHAP values have only one class output (ndim=3, shape[0]=1), using index 0.")
            else: # pragma: no cover
                shap_analyze_logger.error(f"      (Error) Unexpected 3D SHAP values shape: {shap_values_calculated.shape}. Cannot determine positive class.")
        else: # pragma: no cover
            shap_analyze_logger.error(f"      (Error) Unexpected SHAP values structure (Type: {type(shap_values_calculated)}, Shape: {getattr(shap_values_calculated, 'shape', 'N/A')}). Cannot plot.")
            return

        if shap_values_positive_class_plot is None: # pragma: no cover
            shap_analyze_logger.error("      (Error) Could not identify SHAP values for positive class plotting.")
            return
        if shap_values_positive_class_plot.shape[1] != len(features): # pragma: no cover
            shap_analyze_logger.error(f"      (Error) SHAP feature dimension mismatch ({shap_values_positive_class_plot.shape[1]} != {len(features)}). Cannot proceed.")
            return
        if shap_values_positive_class_plot.shape[0] != X_shap.shape[0]: # pragma: no cover
            shap_analyze_logger.error(f"      (Error) SHAP sample dimension mismatch ({shap_values_positive_class_plot.shape[0]} != {X_shap.shape[0]}). Cannot proceed.")
            return

        # Plot SHAP summary (bar plot)
        shap_analyze_logger.info("      Creating SHAP Summary Plot (bar type)...")
        shap_plot_path_bar = os.path.join(output_dir, f"shap_summary_{model_type}_bar{fold_suffix}.png")
        plt.figure() # Create new figure
        try:
            shap.summary_plot(shap_values_positive_class_plot, X_shap, plot_type="bar", show=False, feature_names=features, max_display=20)  # type: ignore
            plt.title(f"SHAP Feature Importance ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_plot_path_bar, dpi=300, bbox_inches="tight")
            shap_analyze_logger.info(f"      (Success) Saved SHAP Plot (Bar): {os.path.basename(shap_plot_path_bar)}")
        except Exception as e_save_bar_shap: # pragma: no cover
            shap_analyze_logger.error(f"      (Error) Failed to create/save SHAP bar plot: {e_save_bar_shap}", exc_info=True)
        finally:
            plt.close() # Close the figure

        # Plot SHAP summary (beeswarm/dot plot)
        shap_analyze_logger.info("      Creating SHAP Summary Plot (beeswarm/dot type)...")
        shap_beeswarm_path_plot = os.path.join(output_dir, f"shap_summary_{model_type}_beeswarm{fold_suffix}.png")
        plt.figure() # Create new figure
        try:
            shap.summary_plot(shap_values_positive_class_plot, X_shap, plot_type="dot", show=False, feature_names=features, max_display=20)  # type: ignore
            plt.title(f"SHAP Feature Summary ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_beeswarm_path_plot, dpi=300, bbox_inches="tight")
            shap_analyze_logger.info(f"      (Success) Saved SHAP Plot (Beeswarm): {os.path.basename(shap_beeswarm_path_plot)}")
        except Exception as e_save_beeswarm_shap: # pragma: no cover
            shap_analyze_logger.error(f"      (Error) Failed to create/save SHAP beeswarm plot: {e_save_beeswarm_shap}", exc_info=True)
        finally:
            plt.close() # Close the figure

    except ImportError: # pragma: no cover
        # This case should ideally be caught by the `if not shap:` check at the beginning
        shap_analyze_logger.error("   (Error) SHAP Error: Could not import shap library components (this should not happen if initial check passed).")
    except Exception as e_shap_analyze: # pragma: no cover
        shap_analyze_logger.error(f"   (Error) Error during SHAP analysis: {e_shap_analyze}", exc_info=True)

# --- Feature Loading Function ---
def load_features_for_model(model_name: str, output_dir_load: str) -> list[str] | None:
    load_feat_logger = logging.getLogger(f"{__name__}.load_features_for_model")
    features_filename_load = f"features_{model_name}.json"
    features_file_path_load = os.path.join(output_dir_load, features_filename_load)
    load_feat_logger.info(f"   (Feature Load) Attempting to load features for '{model_name}' from: {features_file_path_load}")

    if not os.path.exists(features_file_path_load): # pragma: no cover
        load_feat_logger.warning(f"   (Warning) Feature file not found for model '{model_name}': {os.path.basename(features_file_path_load)}")
        # Fallback to main features if specific model's features are not found (and it's not 'main' itself)
        main_features_path_load = os.path.join(output_dir_load, "features_main.json")
        if model_name != 'main' and os.path.exists(main_features_path_load):
            load_feat_logger.info(f"      (Fallback) Loading features from 'features_main.json' instead.")
            features_file_path_load = main_features_path_load
        else:
            load_feat_logger.error(f"      (Fallback Failed) Main feature file also not found or was requested ('{os.path.basename(main_features_path_load)}').")
            return None # No features found
    try:
        with open(features_file_path_load, 'r', encoding='utf-8') as f_load:
            features_loaded = json.load(f_load)
        if isinstance(features_loaded, list) and all(isinstance(feat, str) for feat in features_loaded):
            load_feat_logger.info(f"      (Success) Loaded {len(features_loaded)} features for model '{model_name}' from '{os.path.basename(features_file_path_load)}'.")
            return features_loaded
        else: # pragma: no cover
            load_feat_logger.error(f"   (Error) Invalid format in feature file: {features_file_path_load}. Expected list of strings.")
            return None
    except json.JSONDecodeError as e_json_load: # pragma: no cover
        load_feat_logger.error(f"   (Error) Failed to decode JSON from feature file '{os.path.basename(features_file_path_load)}': {e_json_load}")
        return None
    except Exception as e_load_feat: # pragma: no cover
        load_feat_logger.error(f"   (Error) Failed to load features for model '{model_name}' from '{os.path.basename(features_file_path_load)}': {e_load_feat}", exc_info=True)
        return None

# --- Model Switcher Function ---
def select_model_for_trade(context: dict, available_models: dict | None = None) -> tuple[str | None, float | None]: # <<< MODIFIED: Return type
    """
    Selects the appropriate AI model ('main', 'spike', 'cluster') based on context.
    Falls back to 'main' if the selected model is invalid or missing.
    Returns (None, None) if no valid model (including 'main') can be selected.
    """
    switcher_logger = logging.getLogger(f"{__name__}.select_model_for_trade")
    selected_model_key_switcher: str | None = 'main'  # Default to 'main'
    confidence_switcher: float | None = None

    cluster_value_switcher = context.get('cluster')
    spike_score_value_switcher = context.get('spike_score', 0.0)

    # Ensure context values are numeric or None
    if not isinstance(cluster_value_switcher, (int, float, np.number)) or pd.isna(cluster_value_switcher):  # type: ignore
        cluster_value_switcher = None # Standardize missing cluster to None
    if not isinstance(spike_score_value_switcher, (int, float, np.number)) or pd.isna(spike_score_value_switcher):  # type: ignore
        spike_score_value_switcher = 0.0 # Default spike score if missing or NaN

    # These thresholds could eventually come from StrategyConfig
    spike_switch_threshold_switcher = 0.6  # Example threshold
    cluster_switch_value_config = 2    # Example cluster value that triggers 'cluster' model

    switcher_logger.debug(f"      (Switcher) Context: SpikeScore={spike_score_value_switcher:.3f}, Cluster={cluster_value_switcher}") # type: ignore

    # Determine initial model selection based on context
    if spike_score_value_switcher > spike_switch_threshold_switcher:  # type: ignore
        selected_model_key_switcher = 'spike'
        confidence_switcher = spike_score_value_switcher  # type: ignore
    elif cluster_value_switcher == cluster_switch_value_config:
        selected_model_key_switcher = 'cluster'
        confidence_switcher = 0.8  # Example confidence for cluster model, can be dynamic
    else:
        selected_model_key_switcher = 'main'
        confidence_switcher = None # Main model might not use 'confidence' in this way

    if available_models is None: # pragma: no cover
        switcher_logger.error("      (Switcher Error) 'available_models' is None. No model can be selected.")
        return None, None # Critical: No models available at all

    # Check if the initially selected model is valid (exists, has model object, has features)
    selected_model_info = available_models.get(selected_model_key_switcher, {})
    is_selected_model_valid = selected_model_info.get('model') is not None and selected_model_info.get('features')

    if not is_selected_model_valid:
        switcher_logger.warning(f"      (Switcher Warning) Initially selected model '{selected_model_key_switcher}' is invalid or missing. Attempting fallback to 'main'.")
        selected_model_key_switcher = 'main' # Fallback to 'main'
        confidence_switcher = None # Reset confidence on fallback

        # Check if 'main' model itself is valid
        main_model_info = available_models.get('main', {})
        is_main_model_valid = main_model_info.get('model') is not None and main_model_info.get('features')

        if not is_main_model_valid: # pragma: no cover
            switcher_logger.critical("      (Switcher CRITICAL) Fallback 'main' model is also invalid or missing. No usable model available.")
            return None, None # <<< MODIFIED: Return (None, None) if 'main' is also unusable >>>
    
    switcher_logger.debug(f"      (Switcher) Final Selected Model: '{selected_model_key_switcher}', Confidence: {confidence_switcher}")
    return selected_model_key_switcher, confidence_switcher

logger.info("Part 7 (Original Part 6): Machine Learning Configuration & Helpers Loaded and Refactored.")
# === END OF PART 7/15 ===
# === START OF PART 8/15 ===
# ==============================================================================
# === PART 8: Model Training Function (v4.9.14 - Full Optuna Integration) ===
# ==============================================================================
# <<< MODIFIED: Implemented full Optuna hyperparameter tuning logic. >>>
# <<< Added objective function for Optuna study. >>>
# <<< Final model parameters are now derived from Optuna's best trial if enabled. >>>

import logging  # Already imported
import os  # Already imported
import time  # Already imported
import json  # Already imported
import pandas as pd  # Already imported
import numpy as np  # Already imported
import traceback  # Already imported
from joblib import dump as joblib_dump  # Already imported
from sklearn.model_selection import train_test_split, TimeSeriesSplit  # Already imported
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score # For Optuna objective
import gc  # For memory management

# Import ML libraries conditionally (assuming they are checked/installed in Part 1)
try:
    from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm  # type: ignore
except ImportError:
    CatBoostClassifier = None  # type: ignore
    Pool = None  # type: ignore
    EShapCalcType = None # type: ignore
    EFeaturesSelectionAlgorithm = None # type: ignore
try:
    import shap
except ImportError:
    shap = None  # type: ignore
try:
    import optuna
    if optuna: # pragma: no cover
        optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce Optuna's default verbosity
except ImportError:
    optuna = None  # type: ignore

# Global model paths (can be overridden by specific calls if needed, but usually set once)
# These are defined in Part 1
# META_CLASSIFIER_PATH, SPIKE_MODEL_PATH, CLUSTER_MODEL_PATH

# --- Meta Model Training Function ---
def train_and_export_meta_model(
    config: 'StrategyConfig',  # type: ignore
    output_dir: str,
    model_purpose: str = 'main',
    trade_log_df_override: pd.DataFrame | None = None,
    trade_log_path: str | None = None,
    m1_data_path: str | None = None,
    model_type_to_train: str = "catboost",
    enable_dynamic_feature_selection_override: bool | None = None,
    feature_selection_method_override: str | None = None,
    prelim_model_params_override: dict | None = None,
    enable_optuna_tuning_override: bool | None = None,
    optuna_n_trials_override: int | None = None,
    optuna_cv_splits_override: int | None = None,
    optuna_metric_override: str | None = None,
    optuna_direction_override: str | None = None,
    drift_observer=None,
    catboost_gpu_ram_part_override: float | None = None,
    optuna_n_jobs_override: int | None = None,
    sample_size_override: int | None = None,
    features_to_drop_before_train_override: list | None = None,
    early_stopping_rounds_override: int | None = None,
    shap_importance_threshold_override: float | None = None,
    permutation_importance_threshold_override: float | None = None
) -> tuple[dict | None, list]:
    train_logger = logging.getLogger(f"{__name__}.train_and_export_meta_model.{model_purpose}")
    start_train_time = time.time()
    train_logger.info(f"\n(Training - v4.9.14 Optuna) เริ่มต้นการ Train Meta Classifier (Purpose: {model_purpose.upper()})...")

    # --- Resolve Parameters ---
    enable_dynamic_feature_selection = enable_dynamic_feature_selection_override if enable_dynamic_feature_selection_override is not None else config.enable_dynamic_feature_selection
    enable_optuna_tuning = enable_optuna_tuning_override if enable_optuna_tuning_override is not None else config.enable_optuna_tuning

    feature_selection_method = feature_selection_method_override if feature_selection_method_override is not None else config.feature_selection_method
    optuna_metric_to_optimize = optuna_metric_override if optuna_metric_override is not None else config.optuna_metric
    optuna_direction_to_optimize = optuna_direction_override if optuna_direction_override is not None else config.optuna_direction
    optuna_n_trials_val = optuna_n_trials_override if optuna_n_trials_override is not None else config.optuna_n_trials
    optuna_cv_splits_val = optuna_cv_splits_override if optuna_cv_splits_override is not None else config.optuna_cv_splits
    catboost_gpu_ram_part_val = catboost_gpu_ram_part_override if catboost_gpu_ram_part_override is not None else config.catboost_gpu_ram_part
    optuna_n_jobs_val = optuna_n_jobs_override if optuna_n_jobs_override is not None else config.optuna_n_jobs
    sample_size_val = sample_size_override if sample_size_override is not None else config.sample_size_train
    early_stopping_rounds_val = early_stopping_rounds_override if early_stopping_rounds_override is not None else config.early_stopping_rounds
    shap_threshold_val = shap_importance_threshold_override if shap_importance_threshold_override is not None else config.shap_importance_threshold
    perm_threshold_val = permutation_importance_threshold_override if permutation_importance_threshold_override is not None else config.permutation_importance_threshold

    features_to_drop_val = features_to_drop_before_train_override if features_to_drop_before_train_override is not None else config.features_to_drop_train
    prelim_model_params_val = prelim_model_params_override if prelim_model_params_override is not None else config.prelim_model_params

    initial_features_list_train = config.meta_classifier_features
    if model_purpose == 'spike' and hasattr(config, 'spike_model_features'):
        initial_features_list_train = config.spike_model_features
    elif model_purpose == 'cluster' and hasattr(config, 'cluster_model_features'):
        initial_features_list_train = config.cluster_model_features
    if not initial_features_list_train: 
        initial_features_list_train = ['Gain_Z', 'ATR_14', 'RSI', 'MACD_hist_smooth', 'Pattern_Label', 'session']
        train_logger.warning(f"Initial feature list for '{model_purpose}' was empty in config, using absolute fallback: {initial_features_list_train}")


    train_logger.info(f"   Model Type: {model_type_to_train}")
    train_logger.info(f"   Sample Size Limit: {sample_size_val}")
    train_logger.info(f"   Features to Drop Before Final Train: {features_to_drop_val}")
    train_logger.info(f"   Dynamic Feature Selection: {'เปิดใช้งาน' if enable_dynamic_feature_selection else 'ปิดใช้งาน'}")
    if enable_dynamic_feature_selection:
        train_logger.info(f"     Method: {feature_selection_method.upper()}")
        train_logger.info(f"     SHAP Threshold: {shap_threshold_val:.4f}")
        train_logger.info(f"     Permutation Threshold: {perm_threshold_val:.4f}")
    train_logger.info(f"   Optuna Tuning: {'เปิดใช้งาน' if enable_optuna_tuning else 'ปิดใช้งาน'}")
    if enable_optuna_tuning:
        train_logger.info(f"     Optuna Trials: {optuna_n_trials_val}, CV Splits: {optuna_cv_splits_val}, Metric: {optuna_metric_to_optimize}, Direction: {optuna_direction_to_optimize}, Jobs: {optuna_n_jobs_val}")
    train_logger.info(f"   Early Stopping Rounds (Final Model): {early_stopping_rounds_val}")

    if not output_dir or not isinstance(output_dir, str):
        train_logger.critical("(Error) ไม่ได้ระบุ output_dir หรือไม่ใช่ string.")
        return None, []
    if not os.path.isdir(output_dir):
        try:
            train_logger.info(f"   สร้าง Output Directory: {output_dir}")
            os.makedirs(output_dir)
        except Exception as e_create_dir:
            train_logger.critical(f"(Error) ไม่สามารถสร้าง Output Directory '{output_dir}': {e_create_dir}", exc_info=True)
            return None, []

    global USE_GPU_ACCELERATION, meta_model_type_used
    if enable_optuna_tuning and optuna is None:
        train_logger.warning("(Warning) ต้องการใช้ Optuna แต่ Library ไม่พร้อมใช้งาน. ปิด Optuna Tuning.")
        enable_optuna_tuning = False
    if model_type_to_train == "catboost" and (CatBoostClassifier is None or Pool is None):
        train_logger.critical("(Error) ต้องการ Train CatBoost แต่ Library ไม่พร้อมใช้งาน. ไม่สามารถดำเนินการต่อได้.")
        return None, []

    task_type_setting_train = 'GPU' if USE_GPU_ACCELERATION else 'CPU'
    train_logger.info(f"   GPU Acceleration Available: {USE_GPU_ACCELERATION}. Setting CatBoost task_type to: '{task_type_setting_train}' (for applicable steps).")
    if task_type_setting_train == 'GPU':
        train_logger.info(f"   CatBoost GPU RAM Part setting: {catboost_gpu_ram_part_val:.2f}")
        train_logger.info(f"   CatBoost Device setting: 0 (assuming single GPU)")

    # --- Load Data ---
    trade_log_df_train = None
    if trade_log_df_override is not None and isinstance(trade_log_df_override, pd.DataFrame):
        if trade_log_df_override.empty: train_logger.error(f"(Error) Trade Log Override for '{model_purpose.upper()}' is empty."); return None, []
        required_log_cols_override_train = ["entry_time", "exit_reason"]; missing_cols_override_train = [col for col in required_log_cols_override_train if col not in trade_log_df_override.columns]
        if missing_cols_override_train: train_logger.error(f"(Error) Trade Log Override missing: {missing_cols_override_train}."); return None, []
        train_logger.info(f"   ใช้ Trade Log ที่ Filter แล้ว (Override) จำนวน {len(trade_log_df_override)} แถว สำหรับ Model Purpose: {model_purpose.upper()}")
        trade_log_df_train = trade_log_df_override.copy()
    elif trade_log_path and isinstance(trade_log_path, str):
        train_logger.info(f"   กำลังโหลด Trade Log (Path: {trade_log_path})")
        try:
            trade_log_df_train = safe_load_csv_auto(trade_log_path); # type: ignore
            if trade_log_df_train is None: raise ValueError("safe_load_csv_auto returned None for trade log.")
            if trade_log_df_train.empty: train_logger.error(f"(Error) Trade Log (Path) for '{model_purpose.upper()}' is empty."); return None, []
            required_log_cols_path_train = ["entry_time", "exit_reason"]; missing_cols_path_train = [col for col in required_log_cols_path_train if col not in trade_log_df_train.columns]
            if missing_cols_path_train: train_logger.error(f"(Error) Trade Log (Path) missing: {missing_cols_path_train}."); return None, []
            train_logger.info(f"   โหลด Trade Log (Path) สำเร็จ ({len(trade_log_df_train)} แถว).")
        except Exception as e_load_log_path: train_logger.error(f"(Error) ไม่สามารถโหลด Trade Log (Path): {e_load_log_path}", exc_info=True); return None, []
    else: train_logger.error("(Error) ไม่ได้รับ Trade Log Override และไม่พบไฟล์ Trade Log Path หรือ Path ไม่ถูกต้อง."); return None, []
    try:
        time_cols_log_train = ["entry_time", "close_time", "BE_Triggered_Time"]
        for col_log_train in time_cols_log_train:
            if col_log_train in trade_log_df_train.columns: trade_log_df_train[col_log_train] = pd.to_datetime(trade_log_df_train[col_log_train], errors='coerce')
        if "entry_time" not in trade_log_df_train.columns or not pd.api.types.is_datetime64_any_dtype(trade_log_df_train["entry_time"]): train_logger.error("(Error) 'entry_time' missing or not datetime."); return None, []
        rows_before_drop_log_train = len(trade_log_df_train); trade_log_df_train.dropna(subset=["entry_time"], inplace=True)
        if len(trade_log_df_train) < rows_before_drop_log_train: train_logger.warning(f"   ลบ {rows_before_drop_log_train - len(trade_log_df_train)} trades ที่มี entry_time ไม่ถูกต้อง.")
        trade_log_df_train["is_tp"] = (trade_log_df_train["exit_reason"].astype(str).str.upper() == "TP").astype(int)
        target_dist_train = trade_log_df_train['is_tp'].value_counts(normalize=True).round(3); train_logger.info(f"   Target (is_tp from Log) Distribution:\n{target_dist_train.to_string()}")
        if len(target_dist_train) < 2: train_logger.warning("   (Warning) Target มีเพียง Class เดียว. Model อาจไม่สามารถ Train ได้อย่างมีความหมาย.")
        if trade_log_df_train.empty: train_logger.error("(Error) ไม่มี Trades ที่ถูกต้องใน Log หลังการประมวลผล."); return None, []
        trade_log_df_train = trade_log_df_train.sort_values("entry_time"); train_logger.info(f"   ประมวลผล Trade Log สำเร็จ ({len(trade_log_df_train)} trades).")
    except Exception as e_proc_log: train_logger.error(f"(Error) เกิดข้อผิดพลาดในการประมวลผล Trade Log: {e_proc_log}", exc_info=True); return None, []

    m1_data_path_to_load = getattr(config, 'm1_data_path_train', m1_data_path if m1_data_path else DATA_FILE_PATH_M1) # type: ignore
    train_logger.info(f"   กำลังโหลด M1 Data: {m1_data_path_to_load}")
    if not os.path.exists(m1_data_path_to_load): train_logger.error(f"(Error) ไม่พบ M1 Data file: {m1_data_path_to_load}"); return None, []
    m1_df_for_train = None
    try:
        m1_df_for_train = safe_load_csv_auto(m1_data_path_to_load); # type: ignore
        if m1_df_for_train is None: raise ValueError("safe_load_csv_auto returned None for M1 data.")
        if m1_df_for_train.empty: train_logger.error("   (Error) M1 Data file is empty."); return None, []
        required_m1_features_train = ["Open", "High", "Low", "Close", "ATR_14"]; missing_m1_feats_train = [f for f in required_m1_features_train if f not in m1_df_for_train.columns]
        if missing_m1_feats_train: train_logger.error(f"(Error) M1 Data is missing: {missing_m1_feats_train}."); return None, []
        train_logger.info("   กำลังเตรียม Index ของ M1 Data..."); m1_df_for_train.index = pd.to_datetime(m1_df_for_train.index, errors='coerce')
        rows_before_drop_m1_train = len(m1_df_for_train); m1_df_for_train = m1_df_for_train[m1_df_for_train.index.notna()]
        if len(m1_df_for_train) < rows_before_drop_m1_train: train_logger.warning(f"   ลบ {rows_before_drop_m1_train - len(m1_df_for_train)} แถวที่มี Index เป็น NaT ใน M1 Data.")
        if not isinstance(m1_df_for_train.index, pd.DatetimeIndex): train_logger.error("   (Error) ไม่สามารถแปลง M1 index เป็น DatetimeIndex."); return None, []
        if m1_df_for_train.empty: train_logger.error("   (Error) M1 DataFrame ว่างเปล่าหลังแปลง/ล้าง Index."); return None, []
        if not m1_df_for_train.index.is_monotonic_increasing: train_logger.info("      Sorting M1 DataFrame index..."); m1_df_for_train = m1_df_for_train.sort_index()
        if m1_df_for_train.index.has_duplicates:
            dup_count_m1_train = m1_df_for_train.index.duplicated().sum(); train_logger.warning(f"   (Warning) พบ Index ซ้ำ {dup_count_m1_train} รายการใน M1 Data. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
            m1_df_for_train = m1_df_for_train[~m1_df_for_train.index.duplicated(keep='first')]
        train_logger.info(f"   โหลดและเตรียม M1 สำเร็จ ({len(m1_df_for_train)} แถว). จำนวน Features เริ่มต้น: {len(m1_df_for_train.columns)}")
    except Exception as e_load_m1: train_logger.error(f"(Error) ไม่สามารถโหลดหรือเตรียม M1 data: {e_load_m1}", exc_info=True); return None, []

    train_logger.info(f"   กำลังเตรียมข้อมูลสำหรับ Meta Model Training (Purpose: {model_purpose.upper()})...")
    merged_df_train = None
    try:
        if not pd.api.types.is_datetime64_any_dtype(trade_log_df_train["entry_time"]):
            train_logger.warning("   Converting trade_log entry_time to datetime again before merge.")
            trade_log_df_train["entry_time"] = pd.to_datetime(trade_log_df_train["entry_time"], errors='coerce'); trade_log_df_train.dropna(subset=["entry_time"], inplace=True)
        if trade_log_df_train.empty: train_logger.error("(Error) ไม่มี Trades ที่มี entry_time ถูกต้องหลังการแปลง (ก่อน Merge)."); return None, []
        if not trade_log_df_train["entry_time"].is_monotonic_increasing: trade_log_df_train = trade_log_df_train.sort_values("entry_time")
        if not isinstance(m1_df_for_train.index, pd.DatetimeIndex): train_logger.error("   (Error) M1 index is not DatetimeIndex before merge."); return None, []
        if not m1_df_for_train.index.is_monotonic_increasing: train_logger.warning("   M1 index was not monotonic, sorting again before merge."); m1_df_for_train = m1_df_for_train.sort_index()
        merged_df_train = pd.merge_asof(trade_log_df_train, m1_df_for_train, left_on="entry_time", right_index=True, direction="backward", tolerance=pd.Timedelta(minutes=15))
        train_logger.info(f"   Merge completed. Shape after merge: {merged_df_train.shape}"); del trade_log_df_train, m1_df_for_train; gc.collect()
        current_initial_features = [f for f in initial_features_list_train if f in merged_df_train.columns]
        if not current_initial_features: train_logger.error("(Error) ไม่มี Features เริ่มต้นที่ใช้ได้ในข้อมูลที่รวมแล้ว (จาก config)."); return None, []
        train_logger.info(f"   Features เริ่มต้นสำหรับการเลือก ({len(current_initial_features)} จาก config): {sorted(current_initial_features)}")
        features_to_check_nan_train = current_initial_features + ["is_tp"]; missing_before_dropna_train = [f for f in features_to_check_nan_train if f not in merged_df_train.columns]
        if missing_before_dropna_train: train_logger.error(f"(Error) Critical: merged_df_train ขาด Features ก่อน dropna: {missing_before_dropna_train}"); return None, []
        rows_before_drop_merge_train = len(merged_df_train); train_logger.info(f"   [NaN Check] ก่อน Drop NaN ใน Merged Data (Features/Target): {rows_before_drop_merge_train} แถว")
        merged_df_train.dropna(subset=features_to_check_nan_train, inplace=True); rows_dropped_merge_train = rows_before_drop_merge_train - len(merged_df_train)
        if rows_dropped_merge_train > 0: train_logger.info(f"   [NaN Check] ลบ {rows_dropped_merge_train} Trades ที่มี Missing Features หรือ NaN ใน Features/Target.")
        if merged_df_train.empty: train_logger.error("(Error) ไม่มีข้อมูลสมบูรณ์หลังการรวมและ Drop NaN."); return None, []
        train_logger.info(f"   (Success) การรวมข้อมูลเสร็จสมบูรณ์ ({len(merged_df_train)} samples before sampling).")
        if sample_size_val is not None and sample_size_val > 0 and sample_size_val < len(merged_df_train):
            train_logger.info(f"   Sampling {sample_size_val} rows from merged data..."); merged_df_train = merged_df_train.sample(n=sample_size_val, random_state=42)
            train_logger.info(f"   (Success) Sampled data size: {len(merged_df_train)} rows.")
    except Exception as e_merge_data:
        train_logger.error(f"(Error) เกิดข้อผิดพลาดระหว่างการรวมข้อมูล: {e_merge_data}", exc_info=True)
        if 'trade_log_df_train' in locals() and trade_log_df_train is not None: del trade_log_df_train
        if 'm1_df_for_train' in locals() and m1_df_for_train is not None: del m1_df_for_train
        if 'merged_df_train' in locals() and merged_df_train is not None: del merged_df_train
        gc.collect(); return None, []

    selected_features_train = current_initial_features
    prelim_model_train = None

    if enable_dynamic_feature_selection and model_type_to_train == "catboost" and CatBoostClassifier and Pool and shap and EShapCalcType and EFeaturesSelectionAlgorithm:
        train_logger.info("\n   --- [Phase 2/B] กำลังดำเนินการ Dynamic Feature Selection ---")
        X_select_train = merged_df_train[current_initial_features].copy()
        y_select_train = merged_df_train["is_tp"]
        cat_feature_names_prelim = [col for col in ['Pattern_Label', 'session'] if col in X_select_train.columns]
        cat_indices_prelim_cpu = []
        if cat_feature_names_prelim:
            train_logger.info(f"      จัดการ Categorical Features (Prelim): {cat_feature_names_prelim}...")
            for cat_col_prelim in cat_feature_names_prelim: X_select_train[cat_col_prelim] = X_select_train[cat_col_prelim].astype(str).fillna("Missing_Prelim")
            try: cat_indices_prelim_cpu = [X_select_train.columns.get_loc(col) for col in cat_feature_names_prelim]; train_logger.debug(f"         Indices for CatBoost (CPU - Prelim): {cat_indices_prelim_cpu}")
            except KeyError as e_key_prelim: train_logger.error(f"      (Error) ไม่พบคอลัมน์ Categorical ใน X_select_train (Prelim): {e_key_prelim}."); cat_feature_names_prelim = []; cat_indices_prelim_cpu = []
        
        numeric_cols_prelim = X_select_train.select_dtypes(include=np.number).columns
        X_select_train[numeric_cols_prelim] = X_select_train[numeric_cols_prelim].replace([np.inf, -np.inf], np.nan).fillna(0)
        if X_select_train.isnull().any().any(): train_logger.warning("NaNs found in X_select_train after fill, this might affect prelim model.")

        if prelim_model_params_val is None:
            prelim_model_params_val = {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 42, 'verbose': 0, 'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3, 'early_stopping_rounds': 50, 'auto_class_weights': 'Balanced'}
        prelim_model_params_val['task_type'] = task_type_setting_train
        if task_type_setting_train == 'GPU': prelim_model_params_val['devices'] = '0'; prelim_model_params_val['gpu_ram_part'] = catboost_gpu_ram_part_val
        
        train_logger.info(f"      Training Preliminary CatBoost model for feature selection (Params: {prelim_model_params_val})...")
        prelim_model_train = CatBoostClassifier(**prelim_model_params_val) # type: ignore
        try:
            prelim_model_train.fit(X_select_train, y_select_train, cat_features=cat_indices_prelim_cpu, eval_set=(X_select_train, y_select_train) if len(X_select_train) > 10 else None) 
            train_logger.info("      Preliminary model trained.")
            if feature_selection_method == 'shap' and shap:
                train_logger.info("         Calculating SHAP values for feature selection...")
                explainer_prelim = shap.TreeExplainer(prelim_model_train) # type: ignore
                shap_values_prelim = explainer_prelim.shap_values(Pool(X_select_train, y_select_train, cat_features=cat_indices_prelim_cpu)) # type: ignore
                selected_features_train = select_top_shap_features(shap_values_prelim, current_initial_features, shap_threshold=shap_threshold_val) # type: ignore
            elif feature_selection_method == 'permutation':
                train_logger.warning("Permutation importance for feature selection not fully implemented yet, using all initial features.")
                selected_features_train = current_initial_features
            else: 
                train_logger.info("         Using CatBoost built-in feature selection (RecursiveByShapValues)...")
                summary_prelim = prelim_model_train.select_features(
                    X_select_train, y_select_train,
                    eval_set=Pool(X_select_train, y_select_train, cat_features=cat_indices_prelim_cpu), # type: ignore
                    features_for_select=list(X_select_train.columns),
                    num_features_to_select=max(5, int(len(current_initial_features) * 0.75)), 
                    steps=3, algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues, # type: ignore
                    shap_calc_type=EShapCalcType.Regular, train_final_model=False, # type: ignore
                    logging_level='Silent'
                )
                selected_features_train = summary_prelim['selected_features_names']
        except Exception as e_prelim_fit: train_logger.error(f"      Error during preliminary model training/selection: {e_prelim_fit}", exc_info=True); selected_features_train = current_initial_features
        if not selected_features_train: train_logger.error("      (Error) Dynamic Feature Selection ไม่เหลือ Features เลย! กลับไปใช้ Features เริ่มต้น."); selected_features_train = current_initial_features
        else: train_logger.info(f"   --- [Phase 2/B] Final Selected Features ({len(selected_features_train)}): {sorted(selected_features_train)} ---")
        del X_select_train, y_select_train, prelim_model_train; gc.collect()
    else:
        train_logger.info("   (Info) ข้าม Dynamic Feature Selection. ใช้ Features เริ่มต้นทั้งหมด.")
        selected_features_train = current_initial_features

    train_logger.info(f"\n   กำลังเตรียมข้อมูล Training สุดท้ายด้วย Features ที่เลือก ({len(selected_features_train)} ตัว)...")
    if not selected_features_train: train_logger.error("(Error) ไม่มี Features ที่ถูกเลือกสำหรับ Training."); return None, []
    missing_final_features_train = [f for f in selected_features_train if f not in merged_df_train.columns]
    if missing_final_features_train: train_logger.error(f"(Error) Features ที่เลือก ({missing_final_features_train}) ไม่พบใน merged_df_train."); return None, []

    X_train_final_prep = merged_df_train[selected_features_train].copy()
    y_train_final_prep = merged_df_train["is_tp"]
    
    numeric_cols_final_prep = X_train_final_prep.select_dtypes(include=np.number).columns
    X_train_final_prep[numeric_cols_final_prep] = X_train_final_prep[numeric_cols_final_prep].replace([np.inf, -np.inf], np.nan).fillna(0)
    if X_train_final_prep.isnull().any().any(): train_logger.warning("NaNs found in X_train_final_prep after fill, this might affect final model.")

    if features_to_drop_val and isinstance(features_to_drop_val, list):
        features_actually_dropped = [f for f in features_to_drop_val if f in X_train_final_prep.columns]
        if features_actually_dropped:
            X_train_final_prep.drop(columns=features_actually_dropped, inplace=True)
            selected_features_train = [f for f in selected_features_train if f not in features_actually_dropped]
            train_logger.info(f"   Dropped features before final train: {features_actually_dropped}. Remaining: {len(selected_features_train)}")
    
    X_train_final_prep = X_train_final_prep.astype('float32', errors='ignore')

    final_model_catboost = None
    final_features_used_for_model = selected_features_train
    best_params_from_optuna = None

    if model_type_to_train == "catboost" and CatBoostClassifier and Pool:
        train_logger.info(f"\n   --- Training Final CatBoost Model (Purpose: {model_purpose.upper()}) ---")
        X_train_cat_hpo, X_val_cat_hpo, y_train_cat_hpo, y_val_cat_hpo = train_test_split(
            X_train_final_prep, y_train_final_prep, test_size=0.2, random_state=42, stratify=y_train_final_prep if y_train_final_prep.nunique() > 1 else None
        )
        cat_feature_names_final_model = [col for col in ['Pattern_Label', 'session'] if col in X_train_cat_hpo.columns]
        cat_indices_cpu_final_model = [X_train_cat_hpo.columns.get_loc(col) for col in cat_feature_names_final_model] if cat_feature_names_final_model else []
        
        if enable_optuna_tuning and optuna:
            train_logger.info("      --- Starting Optuna Hyperparameter Optimization ---")
            def objective(trial: optuna.trial.Trial) -> float:
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 3000, step=100),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'depth': trial.suggest_int('depth', 3, 8),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
                    'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255]),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                    'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                    'od_wait': trial.suggest_int('od_wait', 10, 50, step=5),
                    'loss_function': 'Logloss', 'eval_metric': optuna_metric_to_optimize,
                    'random_seed': 42, 'verbose': 0, 'auto_class_weights': 'Balanced',
                    'task_type': task_type_setting_train
                }
                if task_type_setting_train == 'GPU': params['devices'] = '0'; params['gpu_ram_part'] = catboost_gpu_ram_part_val

                cv_scores = []
                tscv_optuna = TimeSeriesSplit(n_splits=optuna_cv_splits_val)
                for train_cv_idx, val_cv_idx in tscv_optuna.split(X_train_cat_hpo):
                    X_train_fold, X_val_fold = X_train_cat_hpo.iloc[train_cv_idx], X_train_cat_hpo.iloc[val_cv_idx]
                    y_train_fold, y_val_fold = y_train_cat_hpo.iloc[train_cv_idx], y_train_cat_hpo.iloc[val_cv_idx]
                    
                    model_cv = CatBoostClassifier(**params) # type: ignore
                    model_cv.fit(X_train_fold, y_train_fold, cat_features=cat_indices_cpu_final_model, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, verbose=0)
                    
                    score_cv = 0.0
                    if optuna_metric_to_optimize == "AUC":
                        preds_proba_cv = model_cv.predict_proba(X_val_fold)[:, 1]
                        score_cv = roc_auc_score(y_val_fold, preds_proba_cv) # type: ignore
                    elif optuna_metric_to_optimize == "Logloss":
                        preds_proba_cv = model_cv.predict_proba(X_val_fold)
                        score_cv = log_loss(y_val_fold, preds_proba_cv) # type: ignore
                    elif optuna_metric_to_optimize == "Accuracy":
                        preds_cv = model_cv.predict(X_val_fold)
                        score_cv = accuracy_score(y_val_fold, preds_cv) # type: ignore
                    else: raise ValueError(f"Unsupported Optuna metric: {optuna_metric_to_optimize}")
                    cv_scores.append(score_cv)
                return np.mean(cv_scores) if cv_scores else 0.0 # Return 0 if no scores (e.g. all folds failed)

            study = optuna.create_study(direction=optuna_direction_to_optimize)
            study.optimize(objective, n_trials=optuna_n_trials_val, n_jobs=optuna_n_jobs_val, show_progress_bar=True)
            best_params_from_optuna = study.best_trial.params
            train_logger.info(f"      Optuna Best Trial {study.best_trial.number}: Value={study.best_trial.value:.4f}")
            train_logger.info(f"      Optuna Best Params: {best_params_from_optuna}")
        
        final_model_params_dict = {
            'loss_function': 'Logloss', 'eval_metric': 'AUC',
            'random_seed': 42, 'verbose': 100, 'auto_class_weights': 'Balanced',
            'early_stopping_rounds': early_stopping_rounds_val,
            'task_type': task_type_setting_train
        }
        if task_type_setting_train == 'GPU': final_model_params_dict['devices'] = '0'; final_model_params_dict['gpu_ram_part'] = catboost_gpu_ram_part_val

        if best_params_from_optuna:
            final_model_params_dict.update(best_params_from_optuna)
            if 'iterations' not in final_model_params_dict: final_model_params_dict['iterations'] = config.catboost_iterations
        else:
            final_model_params_dict['iterations'] = config.catboost_iterations
            final_model_params_dict['learning_rate'] = config.catboost_learning_rate
            final_model_params_dict['depth'] = config.catboost_depth
            final_model_params_dict['l2_leaf_reg'] = config.catboost_l2_leaf_reg
        
        train_logger.info(f"      Training final CatBoost model with params: {final_model_params_dict}")
        final_model_catboost = CatBoostClassifier(**final_model_params_dict) # type: ignore
        eval_pool_final = Pool(X_val_cat_hpo, label=y_val_cat_hpo, cat_features=cat_indices_cpu_final_model) if not X_val_cat_hpo.empty else None
        final_model_catboost.fit(X_train_cat_hpo, y_train_cat_hpo, cat_features=cat_indices_cpu_final_model, eval_set=eval_pool_final)
        meta_model_type_used = final_model_catboost.__class__.__name__
        
        train_logger.info("      Final model trained. Performing quality checks...")
        check_model_overfit(final_model_catboost, X_train_cat_hpo, y_train_cat_hpo, X_val_cat_hpo, y_val_cat_hpo, metric="AUC")
        if shap and EShapCalcType:
            analyze_feature_importance_shap(final_model_catboost, "CatBoostClassifier", X_val_cat_hpo, final_features_used_for_model, output_dir, fold_idx=None)
        
        trained_models_dict = {"catboost": final_model_catboost}
        del X_train_cat_hpo, X_val_cat_hpo, y_train_cat_hpo, y_val_cat_hpo, eval_pool_final; gc.collect()
    else:
        trained_models_dict = {}

    train_logger.info(f"\n   --- Saving Final Model (Purpose: {model_purpose.upper()}) ---")
    saved_model_paths_dict = {}
    if not trained_models_dict:
        train_logger.warning("   (Warning) ไม่มี Models ที่ Train สำเร็จให้ Save.")
        return None, selected_features_train

    for model_name_to_save, model_obj_to_save in trained_models_dict.items():
        if model_name_to_save == "catboost":
            if model_purpose == 'main': model_filename_to_save = config.meta_classifier_filename
            elif model_purpose == 'spike': model_filename_to_save = config.spike_model_filename
            elif model_purpose == 'cluster': model_filename_to_save = config.cluster_model_filename
            else: model_filename_to_save = f"meta_classifier_{model_purpose}.pkl"
            model_path_to_save = os.path.join(output_dir, model_filename_to_save)
            try:
                joblib_dump(model_obj_to_save, model_path_to_save)
                train_logger.info(f"      (Success) Saved Final {model_name_to_save.upper()} (Purpose: {model_purpose.upper()}): {model_path_to_save}")
                saved_model_paths_dict[model_purpose] = model_path_to_save
            except Exception as e_save_final_model:
                train_logger.error(f"      (Error) Failed to save Final {model_name_to_save.upper()} (Purpose: {model_purpose.upper()}): {e_save_final_model}", exc_info=True)
        else:
            train_logger.warning(f"   (Warning) ข้ามการ Save สำหรับ Model Type ที่ไม่คาดคิด: {model_name_to_save}")

    features_filename_to_save = f"features_{model_purpose}.json"
    features_file_path_to_save = os.path.join(output_dir, features_filename_to_save)
    try:
        train_logger.info(f"   Saving final selected features ({len(final_features_used_for_model)}) for '{model_purpose}' to: {features_file_path_to_save}")
        with open(features_file_path_to_save, 'w', encoding='utf-8') as f_save_feat:
            json.dump(final_features_used_for_model, f_save_feat, indent=4, default=simple_converter) # type: ignore
        train_logger.info(f"   (Success) Saved final features list for '{model_purpose}'.")
    except Exception as e_save_final_feat:
        train_logger.error(f"   (Error) Failed to save final features list for '{model_purpose}': {e_save_final_feat}", exc_info=True)

    end_train_time_func = time.time()
    train_logger.info(f"(Finished - v4.9.14 Optuna) Meta Classifier Training (Purpose: {model_purpose.upper()}) complete in {end_train_time_func - start_train_time:.2f} seconds.")
    if 'merged_df_train' in locals() and merged_df_train is not None: del merged_df_train
    if 'X_train_final_prep' in locals() and X_train_final_prep is not None: del X_train_final_prep
    if 'y_train_final_prep' in locals() and y_train_final_prep is not None: del y_train_final_prep
    gc.collect()
    return saved_model_paths_dict, final_features_used_for_model

logger.info("Part 8 (Original Part 7): Model Training Function Loaded and Refactored with Optuna Integration.")
# === END OF PART 8/15 ===

# === START OF PART 9/15 ===
# ==============================================================================
# === PART 9: Backtesting Engine (v4.9.23 - Added TSL/BE Helpers & _check_kill_switch) ===
# ==============================================================================
# <<< MODIFIED: is_entry_allowed now handles (None, None) from model_switcher. >>>
# <<< MODIFIED: run_backtest_simulation_v34 correctly processes potentially None model_key. >>>
# <<< MODIFIED: Changed type hint for model_switcher_func to Optional[Callable]. >>>
# <<< MODIFIED: [Patch] Added new function calculate_lot_by_fund_mode. >>>
# <<< MODIFIED: [Patch] Added new function adjust_lot_tp2_boost. >>>
# <<< MODIFIED: [Patch] Added new function dynamic_tp2_multiplier. >>>
# <<< MODIFIED: [Patch] Added new function spike_guard_blocked. >>>
# <<< MODIFIED: [Patch] Added new function is_reentry_allowed. >>>
# <<< MODIFIED: [Patch] Added new function adjust_lot_recovery_mode. >>>
# <<< MODIFIED: [Patch] Added new function check_margin_call. >>>
# <<< MODIFIED: [Patch] Added new function _check_kill_switch. >>>
# <<< MODIFIED: [Patch] Added new function get_adaptive_tsl_step. >>>
# <<< MODIFIED: [Patch] Added new function update_trailing_sl. >>>
# <<< MODIFIED: [Patch] Added new function maybe_move_sl_to_be. >>>
# <<< MODIFIED: [Patch] run_backtest_simulation_v34 now calls new TSL/BE helpers and _check_kill_switch. >>>
# <<< MODIFIED: [Patch] Updated dynamic_tp2_multiplier, spike_guard_blocked, is_reentry_allowed, adjust_lot_tp2_boost to use getattr for config access. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.1] Integrated all new helper functions into run_backtest_simulation_v34 and refined logic. >>>

import logging  # Already imported
import pandas as pd  # Already imported
import numpy as np  # Already imported
import random  # Already imported
import time  # Already imported
from collections import defaultdict  # Already imported
import gc  # For memory management # Already imported
import math  # For math.isclose # Already imported
import os  # For export functions
import json  # For export functions
from datetime import datetime  # For timestamp in export filenames
from typing import Optional, Callable, Any, Dict, List, Tuple # Ensure this is imported, typically in Part 1

# StrategyConfig, RiskManager, TradeManager are defined in Part 3
# Helper functions like get_session_tag, should_exit_due_to_holding, safe_set_datetime are in other parts
# select_model_for_trade is in Part 7
# CatBoostClassifier is imported in Part 1 (conditionally)

# Ensure tqdm is available (imported in Part 1)
try:
    from tqdm.notebook import tqdm
except ImportError: # pragma: no cover
    tqdm = None  # type: ignore

# --- Order Class ---
class Order:
    def __init__(self, entry_idx: Any, entry_time: pd.Timestamp, entry_price: float, original_lot: float, lot_size: float,
                 original_sl_price: float, sl_price: float, tp_price: float, tp1_price: float,
                 entry_bar_count: int, side: str, m15_trend_zone: str, trade_tag: str,
                 signal_score: float, trade_reason: str, session: str, pattern_label_entry: str,
                 is_reentry: bool, is_forced_entry: bool, meta_proba_tp: Optional[float], meta2_proba_tp: Optional[float],
                 atr_at_entry: float, equity_before_open: float, entry_gain_z: Optional[float], entry_macd_smooth: Optional[float],
                 entry_candle_ratio: Optional[float], entry_adx: Optional[float], entry_volatility_index: Optional[float],
                 risk_mode_at_entry: str, use_trailing_for_tp2: bool, trailing_start_price: Optional[float],
                 trailing_step_r: Optional[float], active_model_at_entry: Optional[str], model_confidence_at_entry: Optional[float],
                 label_suffix: str, config_at_entry: 'StrategyConfig'):  # type: ignore
        self.entry_idx = entry_idx
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.original_lot = original_lot
        self.lot = lot_size
        self.original_sl_price = original_sl_price
        self.sl_price = sl_price
        self.tp_price = tp_price # This is TP2
        self.tp1_price = tp1_price
        self.entry_bar_count = entry_bar_count
        self.side = side
        self.m15_trend_zone = m15_trend_zone
        self.trade_tag = trade_tag
        self.signal_score = signal_score
        self.trade_reason = trade_reason
        self.session = session
        self.pattern_label_entry = pattern_label_entry
        self.is_reentry = is_reentry
        self.is_forced_entry = is_forced_entry
        self.meta_proba_tp = meta_proba_tp
        self.meta2_proba_tp = meta2_proba_tp
        self.atr_at_entry = atr_at_entry
        self.equity_before_open = equity_before_open
        self.entry_gain_z = entry_gain_z
        self.entry_macd_smooth = entry_macd_smooth
        self.entry_candle_ratio = entry_candle_ratio
        self.entry_adx = entry_adx
        self.entry_volatility_index = entry_volatility_index
        self.risk_mode_at_entry = risk_mode_at_entry
        self.active_model_at_entry = active_model_at_entry
        self.model_confidence_at_entry = model_confidence_at_entry
        self.label_suffix = label_suffix
        self.config_at_entry = config_at_entry

        self.closed: bool = False
        self.closed_by_killswitch: bool = False
        self.exit_reason_internal: Optional[str] = None
        self.holding_bars: int = 0
        self.be_triggered: bool = False
        self.be_triggered_time: pd.Timestamp = pd.NaT
        self.tsl_activated: bool = False
        self.trailing_sl_price: float = sl_price # Initial TSL price is the SL price
        self.peak_since_tsl_activation: Optional[float] = entry_price if side == "BUY" else np.nan # Initialize with entry price for BUY
        self.trough_since_tsl_activation: Optional[float] = entry_price if side == "SELL" else np.nan # Initialize with entry price for SELL
        self.reached_tp1: bool = False
        self.partial_tp_processed_levels: set[int] = set()

        self.use_trailing_for_tp2: bool = use_trailing_for_tp2
        self.trailing_start_price: Optional[float] = trailing_start_price # Price at which TSL for TP2 activates
        self.trailing_tp_price: float = tp_price # This is the original TP2 price, TSL might adjust SL towards it
        self.peak_since_ttp2_activation: Optional[float] = np.nan # For TSL on TP2
        self.trough_since_ttp2_activation: Optional[float] = np.nan # For TSL on TP2
        self.trailing_step_r: Optional[float] = trailing_step_r # R-multiple for TSL step, can be adaptive

        self.sl_distance_points: float = abs(self.entry_price - self.original_sl_price) * 10.0 if pd.notna(self.entry_price) and pd.notna(self.original_sl_price) else 0.0
        self.volatility_ratio: float = 1.0 # For adaptive TSL logging

    def to_dict(self) -> Dict[str, Any]:
        log_dict = self.__dict__.copy()
        log_dict.pop('config_at_entry', None) # Don't log the full config object
        return log_dict

# --- Helper function to predefine result columns ---
def _predefine_result_columns_for_simulation(df: pd.DataFrame, label_suffix: str) -> pd.DataFrame:
    predefine_logger = logging.getLogger(f"{__name__}._predefine_result_columns_for_simulation")
    predefine_logger.debug(f"Predefining result columns in df_sim for label_suffix: {label_suffix}")

    df[f"Order_Opened{label_suffix}"] = pd.Series(False, index=df.index, dtype='bool')
    df[f"Lot_Size{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Entry_Price_Actual{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"SL_Price_Actual{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"TP_Price_Actual{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64') # This is TP2
    df[f"TP1_Price_Actual{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"ATR_At_Entry{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Equity_Before_Open{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Is_Reentry{label_suffix}"] = pd.Series(False, index=df.index, dtype='bool')
    df[f"Forced_Entry{label_suffix}"] = pd.Series(False, index=df.index, dtype='bool')
    df[f"Meta_Proba_TP{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Meta2_Proba_TP{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Entry_Gain_Z{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Entry_MACD_Smooth{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Entry_Candle_Ratio{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Entry_ADX{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Entry_Volatility_Index{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Active_Model{label_suffix}"] = pd.Series("NONE", index=df.index, dtype='object')
    df[f"Model_Confidence{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Order_Closed_Time{label_suffix}"] = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    df[f"PnL_Realized_USD{label_suffix}"] = pd.Series(0.0, index=df.index, dtype='float64')
    df[f"Commission_USD{label_suffix}"] = pd.Series(0.0, index=df.index, dtype='float64')
    df[f"Spread_Cost_USD{label_suffix}"] = pd.Series(0.0, index=df.index, dtype='float64')
    df[f"Slippage_USD{label_suffix}"] = pd.Series(0.0, index=df.index, dtype='float64')
    df[f"Exit_Reason_Actual{label_suffix}"] = pd.Series("NONE", index=df.index, dtype='object')
    df[f"Exit_Price_Actual{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"PnL_Points_Actual{label_suffix}"] = pd.Series(0.0, index=df.index, dtype='float64')
    df[f"BE_Triggered_Time{label_suffix}"] = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    df[f"Equity_Realistic{label_suffix}"] = pd.Series(np.nan, index=df.index, dtype='float64')
    df[f"Active_Order_Count{label_suffix}"] = pd.Series(0, index=df.index, dtype='int64')
    df[f"Max_Drawdown_At_Point{label_suffix}"] = pd.Series(0.0, index=df.index, dtype='float64')
    df[f"Risk_Mode{label_suffix}"] = pd.Series("normal", index=df.index, dtype='object')
    predefine_logger.debug(f"Finished predefining {len(df.columns)} columns for {label_suffix}.")
    return df

# --- Backtesting Helper Functions ---

# <<< MODIFIED: [Patch] Added new function calculate_lot_by_fund_mode >>>
def calculate_lot_by_fund_mode(
    config: 'StrategyConfig',  # type: ignore
    mm_mode: str,
    risk_pct: float,
    equity: float,
    atr_val: float, # ATR at entry, used for SL calculation
    sl_delta_price: float # SL distance in price points (e.g., 1.5 * ATR)
) -> float:
    lot_calc_logger = logging.getLogger(f"{__name__}.calculate_lot_by_fund_mode")
    lot_calc_logger.debug(
        f"Calculating lot: MM Mode='{mm_mode}', Risk%={risk_pct:.4f}, Equity=${equity:.2f}, "
        f"ATR={atr_val:.5f}, SL Delta Price={sl_delta_price:.5f}"
    )

    min_lot = config.min_lot
    max_lot = config.max_lot
    point_value_for_min_lot = config.point_value # This is the USD value of 1 point move for 0.01 lot

    if equity <= 0:
        lot_calc_logger.warning("Equity is zero or negative. Returning min_lot.")
        return min_lot
    if sl_delta_price < 1e-5: # Avoid division by zero if SL is too tight
        lot_calc_logger.warning(f"SL delta price is near zero ({sl_delta_price:.5f}). Returning min_lot.")
        return min_lot
    if point_value_for_min_lot < 1e-9: # pragma: no cover
        lot_calc_logger.error("config.point_value is near zero. Cannot calculate lot size. Returning min_lot.")
        return min_lot

    risk_usd_for_trade = equity * risk_pct
    lot_calc_logger.debug(f"   Risk USD for trade: ${risk_usd_for_trade:.2f}")

    # Calculate how much USD is at risk for one minimum lot (0.01)
    # e.g., if SL is 3 price points (30 pips), and point_value for 0.01 lot is $0.1, then risk is 3.0 * $0.1 = $0.3
    total_risk_usd_for_one_min_lot = sl_delta_price * point_value_for_min_lot
    lot_calc_logger.debug(f"   Total risk in USD for one min_lot ({min_lot:.2f}) with SL ${sl_delta_price:.2f}: ${total_risk_usd_for_one_min_lot:.5f}")

    if total_risk_usd_for_one_min_lot < 1e-9: # Avoid division by zero
        lot_calc_logger.warning(f"Total risk for one min_lot is near zero. SL might be too tight or point_value incorrect. Returning min_lot.")
        return min_lot

    # How many 0.01 lot units can we afford given the risk_usd_for_trade
    num_min_lots_can_afford = risk_usd_for_trade / total_risk_usd_for_one_min_lot
    raw_lot = num_min_lots_can_afford * min_lot
    lot_calc_logger.debug(f"   Raw lot size calculated (num_min_lots * min_lot): {raw_lot:.4f}")

    # Adjust based on MM mode (conservative, aggressive, balanced)
    adjusted_lot = raw_lot
    conservative_mult = 0.75 # Could be from config
    aggressive_mult = 1.25   # Could be from config

    if mm_mode == "conservative":
        adjusted_lot *= conservative_mult
        lot_calc_logger.debug(f"   Adjusted lot (conservative *{conservative_mult}): {adjusted_lot:.4f}")
    elif mm_mode == "aggressive":
        adjusted_lot *= aggressive_mult
        lot_calc_logger.debug(f"   Adjusted lot (aggressive *{aggressive_mult}): {adjusted_lot:.4f}")
    elif mm_mode == "balanced":
        lot_calc_logger.debug(f"   Adjusted lot (balanced, no change from raw): {adjusted_lot:.4f}")
    else: # pragma: no cover
        lot_calc_logger.warning(f"Unknown mm_mode '{mm_mode}', using 'balanced' (raw lot).")

    # Ensure lot size is a multiple of min_lot (e.g., 0.01, 0.02, ...)
    if adjusted_lot < min_lot:
        final_lot = min_lot
    else:
        # Round down to the nearest min_lot increment
        num_min_lot_units = math.floor(adjusted_lot / min_lot)
        final_lot = num_min_lot_units * min_lot

    lot_calc_logger.debug(f"   Lot after rounding down to min_lot multiple: {final_lot:.4f}")

    # Apply min/max lot constraints
    final_lot = max(min_lot, final_lot)
    final_lot = min(max_lot, final_lot)
    lot_calc_logger.debug(f"   Lot after min/max capping: {final_lot:.4f}")

    # Final rounding to typical lot precision (e.g., 2 decimal places)
    final_lot = round(final_lot, 2)

    lot_calc_logger.info(f"   [LotCalc] Final Lot Size: {final_lot:.2f} (MM Mode: '{mm_mode}', Risk: {risk_pct:.2%}, Equity: ${equity:.0f})")
    return final_lot
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function adjust_lot_tp2_boost >>>
def adjust_lot_tp2_boost(
    config: 'StrategyConfig',  # type: ignore
    trade_history_list: List[str], # List of exit reasons from recent trades
    base_lot: float
) -> float:
    boost_logger = logging.getLogger(f"{__name__}.adjust_lot_tp2_boost")
    boost_logger.debug(f"Adjusting lot for TP2 boost. Base lot: {base_lot:.2f}, History length: {len(trade_history_list)}")

    # <<< MODIFIED: [Patch] Use getattr for config access >>>
    LOOKBACK_TRADES_FOR_BOOST = getattr(config, 'tp2_boost_lookback_trades', 3)
    TP_COUNT_FOR_BOOST = getattr(config, 'tp2_boost_tp_count_threshold', 2)
    BOOST_MULTIPLIER = getattr(config, 'tp2_boost_multiplier', 1.10)

    if not trade_history_list or len(trade_history_list) < LOOKBACK_TRADES_FOR_BOOST:
        boost_logger.debug("   Not enough trade history for TP2 boost. Returning base lot.")
        return base_lot

    recent_trades = trade_history_list[-LOOKBACK_TRADES_FOR_BOOST:]
    tp_count = 0
    for reason in recent_trades:
        # Ensure 'reason' is a string and check for "TP" while excluding "PARTIAL TP"
        if isinstance(reason, str) and "TP" in reason.upper() and "PARTIAL" not in reason.upper():
            tp_count += 1

    boost_logger.debug(f"   Recent {LOOKBACK_TRADES_FOR_BOOST} trades: {recent_trades}. Full TP count: {tp_count}")

    boosted_lot = base_lot
    if tp_count >= TP_COUNT_FOR_BOOST:
        boosted_lot = base_lot * BOOST_MULTIPLIER
        boost_logger.info(f"   [TP2 Boost Applied] Conditions met ({tp_count} TPs in last {LOOKBACK_TRADES_FOR_BOOST}). Lot boosted from {base_lot:.2f} to {boosted_lot:.2f} (Multiplier: {BOOST_MULTIPLIER:.2f})")
    else: # pragma: no cover
        boost_logger.debug("   TP2 boost conditions not met. Returning base lot.")
        return base_lot # Return base_lot if conditions not met

    # Apply min/max lot constraints and rounding
    final_boosted_lot = min(boosted_lot, config.max_lot)
    final_boosted_lot = max(final_boosted_lot, config.min_lot)
    final_boosted_lot = round(final_boosted_lot, 2)

    if not math.isclose(final_boosted_lot, base_lot): # Log only if there was an actual change after capping
        boost_logger.info(f"   Final boosted lot after capping and rounding: {final_boosted_lot:.2f}")
    else: # pragma: no cover
        boost_logger.debug(f"   Final lot after potential boost (no change or capped): {final_boosted_lot:.2f}")

    return final_boosted_lot
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function dynamic_tp2_multiplier >>>
def dynamic_tp2_multiplier(
    config: 'StrategyConfig',  # type: ignore
    current_atr: Optional[float], # Current bar's ATR_14
    avg_atr: Optional[float]      # Current bar's ATR_14_Rolling_Avg
) -> float:
    tp2_mult_logger = logging.getLogger(f"{__name__}.dynamic_tp2_multiplier")
    base_multiplier = config.base_tp_multiplier

    if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr is None or avg_atr < 1e-9: # type: ignore
        tp2_mult_logger.debug(f"Invalid ATR values (current: {current_atr}, avg: {avg_atr}). Returning base TP multiplier: {base_multiplier:.2f}")
        return base_multiplier

    # <<< MODIFIED: [Patch] Use getattr for config access >>>
    vol_high_ratio = getattr(config, 'tp2_dynamic_vol_high_ratio', getattr(config, 'adaptive_tsl_high_vol_ratio', 1.8))
    vol_low_ratio = getattr(config, 'tp2_dynamic_vol_low_ratio', getattr(config, 'adaptive_tsl_low_vol_ratio', 0.75))
    high_vol_boost_factor = getattr(config, 'tp2_dynamic_high_vol_boost', 1.2)
    low_vol_reduce_factor = getattr(config, 'tp2_dynamic_low_vol_reduce', 0.8)
    min_tp_multiplier_cfg = getattr(config, 'tp2_dynamic_min_multiplier', base_multiplier * 0.5)
    max_tp_multiplier_cfg = getattr(config, 'tp2_dynamic_max_multiplier', base_multiplier * 2.0)

    ratio = current_atr / avg_atr # type: ignore
    final_multiplier = base_multiplier

    if ratio > vol_high_ratio:
        final_multiplier = base_multiplier * high_vol_boost_factor
        tp2_mult_logger.debug(f"High volatility detected (ratio: {ratio:.2f} > {vol_high_ratio:.2f}). Boosting TP2 multiplier to {final_multiplier:.2f} (Base: {base_multiplier:.2f})")
    elif ratio < vol_low_ratio:
        final_multiplier = base_multiplier * low_vol_reduce_factor
        tp2_mult_logger.debug(f"Low volatility detected (ratio: {ratio:.2f} < {vol_low_ratio:.2f}). Reducing TP2 multiplier to {final_multiplier:.2f} (Base: {base_multiplier:.2f})")
    else: # pragma: no cover
        tp2_mult_logger.debug(f"Normal volatility (ratio: {ratio:.2f}). Using base TP2 multiplier: {base_multiplier:.2f}")

    # Apply min/max caps
    final_multiplier = max(min_tp_multiplier_cfg, min(final_multiplier, max_tp_multiplier_cfg))
    tp2_mult_logger.info(f"   [TP2 Mult] Final Dynamic TP2 Multiplier: {final_multiplier:.3f} (Base: {base_multiplier:.2f}, ATR Ratio: {ratio:.2f})")
    return round(final_multiplier, 3)
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function spike_guard_blocked >>>
def spike_guard_blocked(
    row_data: pd.Series,
    session_tag: str,
    config: 'StrategyConfig'  # type: ignore
) -> bool:
    """
    Checks if a trade should be blocked by the Spike Guard, primarily during London session.
    """
    sg_logger = logging.getLogger(f"{__name__}.spike_guard_blocked")

    # <<< MODIFIED: [Patch] Use getattr for config access >>>
    if not getattr(config, 'enable_spike_guard', True):
        return False

    is_london_session_active = "london" in session_tag.lower()

    if is_london_session_active:
        spike_score = row_data.get('spike_score', 0.0)
        pattern_label = str(row_data.get('Pattern_Label', 'Normal')) # Ensure it's a string
        threshold = getattr(config, 'spike_guard_score_threshold', 0.75)
        allowed_patterns = getattr(config, 'spike_guard_london_patterns', ["Breakout", "StrongTrend"])

        sg_logger.debug(f"Spike Guard (London Active): Score={spike_score:.2f} (Thresh={threshold:.2f}), Pattern='{pattern_label}', AllowedPatterns={allowed_patterns}")

        if not isinstance(allowed_patterns, list): # pragma: no cover
            sg_logger.warning(f"spike_guard_london_patterns is not a list in config: {allowed_patterns}. Spike Guard may not function as expected.")
            return False # Fail safe if config is malformed

        if spike_score > threshold and pattern_label in allowed_patterns:
            sg_logger.info(f"   [Spike Guard BLOCKED] Conditions met: SpikeScore ({spike_score:.2f}) > Threshold ({threshold:.2f}) AND Pattern ('{pattern_label}') in {allowed_patterns}.")
            return True
        else: # pragma: no cover
            sg_logger.debug("   Spike Guard (London Active): Conditions not met for blocking.")
    else: # pragma: no cover
        sg_logger.debug(f"   Spike Guard: Not London session ('{session_tag}'). No blocking by this rule.")

    return False
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function is_reentry_allowed >>>
def is_reentry_allowed(
    config: 'StrategyConfig',  # type: ignore
    row_data: pd.Series, # Current bar data, row_data.name is the timestamp
    side: str, # "BUY" or "SELL"
    active_orders: List[Order], # List of Order objects
    bars_since_last_trade: int,
    last_tp_time_for_side: pd.Timestamp, # Time of last TP for the current side
    meta_proba_tp: Optional[float] # Optional ML probability
) -> bool:
    """
    Checks if a re-entry trade is allowed based on configuration and current state.
    """
    reentry_logger = logging.getLogger(f"{__name__}.is_reentry_allowed.{side}")

    if not config.use_reentry:
        reentry_logger.debug("Re-entry disabled in config.")
        return False

    if bars_since_last_trade < config.reentry_cooldown_bars:
        reentry_logger.debug(f"Re-entry blocked: Cooldown active. Bars since last trade ({bars_since_last_trade}) < Cooldown ({config.reentry_cooldown_bars}).")
        return False

    if meta_proba_tp is not None and pd.notna(meta_proba_tp) and meta_proba_tp < config.reentry_min_proba_thresh:
        reentry_logger.debug(f"Re-entry blocked: Meta proba ({meta_proba_tp:.3f}) < Threshold ({config.reentry_min_proba_thresh:.3f}).")
        return False

    # Check if there's already an active order for the same side
    for order in active_orders:
        if order.side == side and not order.closed: # pragma: no cover
            reentry_logger.debug(f"Re-entry blocked: Active order already exists for side {side}.")
            return False

    # Optional: Cooldown after a TP on the same side
    # <<< MODIFIED: [Patch] Use getattr for config access >>>
    reentry_cooldown_after_tp_min = getattr(config, 'reentry_cooldown_after_tp_minutes', 0)
    if reentry_cooldown_after_tp_min > 0 and pd.notna(last_tp_time_for_side):
        current_time_reentry = row_data.name # Assuming row_data.name is the current timestamp
        if pd.isna(current_time_reentry): # pragma: no cover
            reentry_logger.warning("Current time (row_data.name) is NaT for re-entry TP cooldown check.")
        elif (current_time_reentry - last_tp_time_for_side).total_seconds() / 60 < reentry_cooldown_after_tp_min: # pragma: no cover
            reentry_logger.debug(f"Re-entry blocked: Cooldown after last TP active for side {side}.")
            return False

    reentry_logger.info(f"Re-entry ALLOWED for side {side} at {row_data.name if hasattr(row_data, 'name') else 'N/A'}.")
    return True
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function adjust_lot_recovery_mode >>>
def adjust_lot_recovery_mode(
    config: 'StrategyConfig',  # type: ignore
    current_lot: float,
    consecutive_losses: int
) -> Tuple[float, str]:
    """
    Adjusts the lot size if in recovery mode.
    Returns the adjusted lot size and the current risk mode ("normal" or "recovery").
    """
    recovery_logger = logging.getLogger(f"{__name__}.adjust_lot_recovery_mode")
    risk_mode = "normal"
    adjusted_lot = current_lot

    if consecutive_losses >= config.recovery_mode_consecutive_losses:
        risk_mode = "recovery"
        adjusted_lot = current_lot * config.recovery_mode_lot_multiplier
        adjusted_lot = max(adjusted_lot, config.min_lot)  # Ensure not below min_lot
        adjusted_lot = round(adjusted_lot, 2)  # Round to standard lot precision
        recovery_logger.info(
            f"[Recovery Mode Active] Consecutive losses ({consecutive_losses}) >= threshold ({config.recovery_mode_consecutive_losses}). "
            f"Lot adjusted from {current_lot:.2f} to {adjusted_lot:.2f} (Multiplier: {config.recovery_mode_lot_multiplier})."
        )
    else: # pragma: no cover
        recovery_logger.debug(
            f"[Normal Mode] Consecutive losses ({consecutive_losses}) < threshold ({config.recovery_mode_consecutive_losses}). "
            f"Lot remains {current_lot:.2f}."
        )
    return adjusted_lot, risk_mode
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function check_margin_call >>>
def check_margin_call(current_equity: float, margin_call_level: float = 0.0) -> bool:
    """
    Checks if the current equity has hit the margin call level.
    A margin_call_level of 0 means any equity <= 0 is a margin call.
    """
    mc_logger = logging.getLogger(f"{__name__}.check_margin_call")
    if current_equity <= margin_call_level: # pragma: no cover
        mc_logger.critical(
            f"[MARGIN CALL] Current equity ({current_equity:.2f}) is at or below margin call level ({margin_call_level:.2f})."
        )
        return True
    return False
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function _check_kill_switch >>>
def _check_kill_switch(
    current_equity: float,
    peak_equity: float,
    kill_switch_dd_threshold_config: float,
    kill_switch_consecutive_losses_config: int,
    current_consecutive_losses: int,
    kill_switch_active_status: bool, # Current status of the kill switch
    current_time: pd.Timestamp, # For logging
    config: 'StrategyConfig',  # type: ignore # Pass full config for future flexibility
    logger_parent: logging.Logger # Use parent logger and create child
) -> Tuple[bool, bool]:
    """
    Checks for Kill Switch conditions (Drawdown or Consecutive Losses).
    Returns (kill_switch_triggered_this_bar, new_kill_switch_active_status).
    """
    ks_logger = logging.getLogger(f"{logger_parent.name}._check_kill_switch")
    ks_logger.debug(
        f"Checking Kill Switch at {current_time}: Equity={current_equity:.2f}, Peak={peak_equity:.2f}, "
        f"DDThresh={kill_switch_dd_threshold_config:.2%}, ConsLossThresh={kill_switch_consecutive_losses_config}, "
        f"CurrentConsLoss={current_consecutive_losses}, ActiveStatus={kill_switch_active_status}"
    )

    if kill_switch_active_status: # If already active, it remains active
        ks_logger.debug("Kill switch already active from previous bar/check.")
        return True, True # Triggered (still), Active (still)

    triggered_by_dd = False
    if kill_switch_dd_threshold_config > 0.0: # Check if DD kill switch is enabled
        if peak_equity <= 1e-9: # Avoid division by zero or near-zero peak
            drawdown = 0.0
            ks_logger.debug("Peak equity is near zero, drawdown calculated as 0.")
        else:
            drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown >= kill_switch_dd_threshold_config: # pragma: no cover
            triggered_by_dd = True
            ks_logger.critical(
                f"[KILL SWITCH - DD] Triggered at {current_time}. "
                f"DD={drawdown:.2%} >= Threshold={kill_switch_dd_threshold_config:.2%}. "
                f"Equity={current_equity:.2f}, Peak={peak_equity:.2f}"
            )
    else: # pragma: no cover
        ks_logger.debug("Drawdown Kill Switch is disabled (threshold <= 0.0).")

    triggered_by_consecutive_loss = False
    if kill_switch_consecutive_losses_config > 0: # Check if consecutive loss kill switch is enabled
        if current_consecutive_losses >= kill_switch_consecutive_losses_config: # pragma: no cover
            triggered_by_consecutive_loss = True
            ks_logger.critical(
                f"[KILL SWITCH - CONSECUTIVE LOSS] Triggered at {current_time}. "
                f"Losses={current_consecutive_losses} >= Threshold={kill_switch_consecutive_losses_config}."
            )
    else: # pragma: no cover
        ks_logger.debug("Consecutive Loss Kill Switch is disabled (threshold <= 0).")

    kill_switch_triggered_this_bar = triggered_by_dd or triggered_by_consecutive_loss
    new_active_status = kill_switch_active_status or kill_switch_triggered_this_bar

    if kill_switch_triggered_this_bar and not kill_switch_active_status: # pragma: no cover
        ks_logger.info(f"Kill Switch newly activated at {current_time} (DD: {triggered_by_dd}, ConsLoss: {triggered_by_consecutive_loss}).")

    ks_logger.debug(f"Kill Switch check result: TriggeredNow={kill_switch_triggered_this_bar}, NewActiveStatus={new_active_status}")
    return kill_switch_triggered_this_bar, new_active_status
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function get_adaptive_tsl_step >>>
def get_adaptive_tsl_step(
    order: Order,
    config: 'StrategyConfig',  # type: ignore
    current_atr: Optional[float], # Current bar's ATR_14
    avg_atr: Optional[float],     # Current bar's ATR_14_Rolling_Avg
    logger_parent: logging.Logger
) -> float:
    """
    Calculates the adaptive trailing stop loss step R-multiple based on volatility.
    Updates order.volatility_ratio and order.trailing_step_r.
    """
    tsl_step_logger = logging.getLogger(f"{logger_parent.name}.get_adaptive_tsl_step.{order.side}.{order.entry_idx}")
    tsl_step_logger.debug(f"Getting adaptive TSL step. current_atr={current_atr}, avg_atr={avg_atr}, Order ATR@Entry={order.atr_at_entry}")

    if current_atr is None or avg_atr is None or pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr < 1e-9:
        tsl_step_logger.debug(f"Invalid ATR for TSL step calculation. Using default: {config.adaptive_tsl_default_step_r}")
        order.volatility_ratio = 1.0  # Default volatility ratio
        order.trailing_step_r = config.adaptive_tsl_default_step_r
        return config.adaptive_tsl_default_step_r

    volatility_ratio = current_atr / avg_atr
    order.volatility_ratio = volatility_ratio  # Store for logging/analysis

    current_tsl_step_r = config.adaptive_tsl_default_step_r
    log_msg_vol = f"VolRatio={volatility_ratio:.2f}."

    if volatility_ratio > config.adaptive_tsl_high_vol_ratio:
        current_tsl_step_r = config.adaptive_tsl_high_vol_step_r
        log_msg_vol += f" High Vol. Step set to {current_tsl_step_r:.2f}R."
    elif volatility_ratio < config.adaptive_tsl_low_vol_ratio:
        current_tsl_step_r = config.adaptive_tsl_low_vol_step_r
        log_msg_vol += f" Low Vol. Step set to {current_tsl_step_r:.2f}R."
    else: # pragma: no cover
        log_msg_vol += f" Normal Vol. Step remains default {current_tsl_step_r:.2f}R."

    tsl_step_logger.debug(log_msg_vol)
    order.trailing_step_r = current_tsl_step_r # Store the actual step R used
    return current_tsl_step_r
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function update_trailing_sl >>>
def update_trailing_sl(
    order: Order,
    config: 'StrategyConfig',  # type: ignore
    current_price_for_tsl: float, # High for BUY, Low for SELL of the current bar
    current_atr: Optional[float], # Current bar's ATR_14
    avg_atr: Optional[float],     # Current bar's ATR_14_Rolling_Avg
    logger_parent: logging.Logger
):
    """
    Updates the trailing stop loss for an active order.
    """
    tsl_logger = logging.getLogger(f"{logger_parent.name}.update_trailing_sl.{order.side}.{order.entry_idx}")

    if not order.tsl_activated or order.closed:
        if not order.tsl_activated: tsl_logger.debug("TSL not activated for this order.")
        if order.closed: tsl_logger.debug("Order is closed, skipping TSL update.")
        return

    # Get the adaptive TSL step R-multiple (this also updates order.trailing_step_r)
    current_tsl_step_r = get_adaptive_tsl_step(order, config, current_atr, avg_atr, tsl_logger)

    if order.atr_at_entry is None or pd.isna(order.atr_at_entry) or order.atr_at_entry < 1e-9:
        tsl_logger.warning(f"atr_at_entry is invalid ({order.atr_at_entry}). Cannot calculate TSL distance.")
        return

    # TSL distance is based on ATR at entry and the current adaptive step R-multiple
    sl_distance_for_tsl = order.atr_at_entry * current_tsl_step_r
    if sl_distance_for_tsl < 1e-9: # Ensure distance is meaningful
        tsl_logger.warning(f"TSL distance is near zero ({sl_distance_for_tsl}). Skipping TSL update.")
        return

    new_potential_sl = np.nan
    can_move_sl = False

    if order.side == "BUY":
        prev_peak = order.peak_since_tsl_activation
        current_peak = max(order.peak_since_tsl_activation if pd.notna(order.peak_since_tsl_activation) else -np.inf, current_price_for_tsl)
        order.peak_since_tsl_activation = current_peak # Update peak

        new_potential_sl = current_peak - sl_distance_for_tsl

        tsl_logger.debug(
            f"BUY TSL Check: CurrentPrice={current_price_for_tsl:.5f}, Peak={current_peak:.5f} (PrevPeak={prev_peak:.5f if pd.notna(prev_peak) else 'N/A'}), "
            f"SLDist={sl_distance_for_tsl:.5f}, NewPotentialSL={new_potential_sl:.5f}, "
            f"CurrentSL={order.sl_price:.5f}, Entry={order.entry_price:.5f}, BE_Triggered={order.be_triggered}"
        )

        # Condition: New SL must be better (higher) than current SL.
        # If BE triggered, new SL must also be above entry.
        # If not BE triggered, new SL can be below entry as long as it's an improvement.
        if new_potential_sl > order.sl_price:
            if order.be_triggered: # If BE is active, TSL must be >= entry price
                can_move_sl = new_potential_sl >= order.entry_price
            else: # If BE not active, any improvement is fine
                can_move_sl = True

    elif order.side == "SELL":
        prev_trough = order.trough_since_tsl_activation
        current_trough = min(order.trough_since_tsl_activation if pd.notna(order.trough_since_tsl_activation) else np.inf, current_price_for_tsl)
        order.trough_since_tsl_activation = current_trough # Update trough

        new_potential_sl = current_trough + sl_distance_for_tsl

        tsl_logger.debug(
            f"SELL TSL Check: CurrentPrice={current_price_for_tsl:.5f}, Trough={current_trough:.5f} (PrevTrough={prev_trough:.5f if pd.notna(prev_trough) else 'N/A'}), "
            f"SLDist={sl_distance_for_tsl:.5f}, NewPotentialSL={new_potential_sl:.5f}, "
            f"CurrentSL={order.sl_price:.5f}, Entry={order.entry_price:.5f}, BE_Triggered={order.be_triggered}"
        )

        if new_potential_sl < order.sl_price:
            if order.be_triggered: # If BE is active, TSL must be <= entry price
                can_move_sl = new_potential_sl <= order.entry_price
            else: # If BE not active, any improvement is fine
                can_move_sl = True

    if can_move_sl and pd.notna(new_potential_sl):
        tsl_logger.info(
            f"[TSL Update] Order {order.entry_idx} ({order.side}): SL moved from {order.sl_price:.5f} to {new_potential_sl:.5f}. "
            f"VolRatio: {order.volatility_ratio:.2f}, StepR: {current_tsl_step_r:.2f} (ATR@Entry: {order.atr_at_entry:.5f})"
        )
        order.trailing_sl_price = new_potential_sl # Store the pure TSL calculation
        order.sl_price = new_potential_sl         # Update the actual SL price
    else: # pragma: no cover
        tsl_logger.debug(
            f"Order {order.entry_idx} ({order.side}): TSL conditions not met to move SL from {order.sl_price:.5f} "
            f"(NewPotentialSL={new_potential_sl:.5f}, CanMoveEval={can_move_sl})."
        )
# <<< END OF MODIFIED [Patch] >>>

# <<< MODIFIED: [Patch] Added new function maybe_move_sl_to_be >>>
def maybe_move_sl_to_be(
    order: Order,
    config: 'StrategyConfig',  # type: ignore
    current_price_for_be: float, # High for BUY, Low for SELL of the current bar
    current_time: pd.Timestamp,
    logger_parent: logging.Logger
):
    """
    Checks if the Stop Loss should be moved to Breakeven based on R-multiple.
    Updates order.sl_price, order.be_triggered, and order.be_triggered_time.
    If TSL is active, also ensures order.trailing_sl_price respects the new BE SL.
    """
    be_logger = logging.getLogger(f"{logger_parent.name}.maybe_move_sl_to_be.{order.side}.{order.entry_idx}")

    if order.be_triggered or order.closed:
        be_logger.debug(f"BE check skipped: Already BE triggered ({order.be_triggered}) or order closed ({order.closed}).")
        return

    # If PTP is configured to move SL to entry and it has already done so (reached_tp1 is True),
    # then this R-multiple based BE movement might be redundant or conflicting.
    # However, if PTP is NOT configured to move SL, then this R-multiple BE is the primary way.
    if config.partial_tp_move_sl_to_entry and order.reached_tp1:
        be_logger.debug("BE check (R-multiple based) skipped: PTP has already moved SL to entry as per config.")
        return

    r_threshold_for_be = config.base_be_sl_r_threshold
    if r_threshold_for_be <= 0: # Ensure R-threshold is positive and meaningful
        be_logger.debug(f"BE check skipped: base_be_sl_r_threshold ({r_threshold_for_be}) is not positive.")
        return

    sl_distance_initial = abs(order.entry_price - order.original_sl_price)
    if sl_distance_initial < 1e-9: # Avoid division by zero or very small SL
        be_logger.warning(f"Initial SL distance is near zero ({sl_distance_initial}). Cannot calculate BE target accurately.")
        return

    price_movement_needed_for_be = sl_distance_initial * r_threshold_for_be
    be_triggered_this_bar = False
    target_price_for_be_trigger = np.nan

    if order.side == "BUY":
        target_price_for_be_trigger = order.entry_price + price_movement_needed_for_be
        if current_price_for_be >= target_price_for_be_trigger:
            be_triggered_this_bar = True
    elif order.side == "SELL":
        target_price_for_be_trigger = order.entry_price - price_movement_needed_for_be
        if current_price_for_be <= target_price_for_be_trigger:
            be_triggered_this_bar = True

    be_logger.debug(
        f"BE Check: CurrentPrice={current_price_for_be:.5f}, TargetForBE={target_price_for_be_trigger:.5f}, "
        f"R_Thresh={r_threshold_for_be:.2f} (from SLDist={sl_distance_initial:.5f}), Entry={order.entry_price:.5f}, CurrentSL={order.sl_price:.5f}"
    )

    if be_triggered_this_bar:
        if not math.isclose(order.sl_price, order.entry_price): # Only move if not already at entry
            be_logger.info(
                f"[BE Triggered (R-Multiple)] Order {order.entry_idx} ({order.side}): Price reached {current_price_for_be:.5f} "
                f"(Target: {target_price_for_be_trigger:.5f}). Moving SL from {order.sl_price:.5f} to Entry {order.entry_price:.5f}."
            )
            order.sl_price = order.entry_price
            order.be_triggered = True
            order.be_triggered_time = current_time

            # If TSL is active, its trailing_sl_price should also respect the new BE SL.
            # This means TSL cannot be worse than the breakeven price.
            if order.tsl_activated:
                if order.side == "BUY" and order.trailing_sl_price < order.entry_price:
                    be_logger.debug(f"  Adjusting TSL due to BE. Old TSL: {order.trailing_sl_price:.5f}, New TSL (at entry): {order.entry_price:.5f}")
                    order.trailing_sl_price = order.entry_price
                elif order.side == "SELL" and order.trailing_sl_price > order.entry_price:
                    be_logger.debug(f"  Adjusting TSL due to BE. Old TSL: {order.trailing_sl_price:.5f}, New TSL (at entry): {order.entry_price:.5f}")
                    order.trailing_sl_price = order.entry_price
        else: # pragma: no cover
            be_logger.debug(f"Order {order.entry_idx}: BE condition (R-Multiple) met, but SL already at entry price. Setting be_triggered=True.")
            order.be_triggered = True # Ensure flag is set even if SL was already there (e.g. by PTP)
            if pd.isna(order.be_triggered_time): order.be_triggered_time = current_time
    else: # pragma: no cover
        be_logger.debug(f"Order {order.entry_idx}: BE condition (R-Multiple) not met.")
# <<< END OF MODIFIED [Patch] >>>


# <<< MODIFIED: is_entry_allowed now handles (None, None) from model_switcher. >>>
def is_entry_allowed(
    config: 'StrategyConfig', risk_manager: 'RiskManager', trade_manager: 'TradeManager',  # type: ignore
    row_data: pd.Series, session_tag: str, consecutive_losses_run: int,
    equity: float, peak_equity: float, initial_capital_segment: float,
    side: str, active_orders: List[Order], bars_since_last_trade: int,
    last_tp_time_for_side: Dict[str, pd.Timestamp], # Changed to dict
    fold_config_override: Dict[str, Any], available_models: Optional[Dict[str, Any]],
    model_switcher_func: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[Optional[str], Optional[float]]]], # <<< MODIFIED: Type Hint
    current_l1_threshold_override: float,
    min_ts_val: pd.Timestamp, label_suffix_entry: str
) -> Tuple[bool, str, str, Optional[float], Optional[str], Optional[float]]:
    entry_allow_logger = logging.getLogger(f"{__name__}.is_entry_allowed.{label_suffix_entry}.{side}")

    if not risk_manager.is_trading_allowed():
        entry_allow_logger.debug(f"Entry blocked: Soft Kill Active by RiskManager.")
        return False, "SOFT_KILL_ACTIVE", "Normal", np.nan, None, np.nan

    if config.enable_spike_guard and spike_guard_blocked(row_data, session_tag, config):
        entry_allow_logger.info(f"Entry blocked by Spike Guard (London) at {row_data.name}")
        return False, "SPIKE_GUARD_LONDON", "Normal", np.nan, None, np.nan

    signal_score_allow = row_data.get('Signal_Score', 0.0)
    entry_long_signal = row_data.get('Entry_Long', 0)
    entry_short_signal = row_data.get('Entry_Short', 0)

    if side == "BUY" and entry_long_signal == 0:
        return False, "NO_VALID_SIGNAL", "Normal", np.nan, None, np.nan
    if side == "SELL" and entry_short_signal == 0:
        return False, "NO_VALID_SIGNAL", "Normal", np.nan, None, np.nan
    if abs(signal_score_allow) < config.min_signal_score_entry:
        return False, f"LOW_BASE_SIGNAL_SCORE ({abs(signal_score_allow):.2f} < {config.min_signal_score_entry:.2f})", "Normal", np.nan, None, np.nan

    meta_proba_tp_allow: Optional[float] = np.nan
    active_model_key_allow: Optional[str] = "SignalOnly"
    model_confidence_allow: Optional[float] = np.nan

    if model_switcher_func and available_models and config.use_meta_classifier:
        context_dict_allow = {
            'cluster': row_data.get('cluster'),
            'spike_score': row_data.get('spike_score', 0.0),
            'session': session_tag,
            'pattern': row_data.get('Pattern_Label')
        }
        selected_model_key, selected_model_confidence = model_switcher_func(context_dict_allow, available_models)

        # <<< MODIFIED: Handle (None, None) from model_switcher >>>
        if selected_model_key is None: # pragma: no cover
            entry_allow_logger.warning(f"[{label_suffix_entry}] No valid model could be selected by switcher. Blocking entry.")
            return False, "NO_VALID_MODEL_AVAILABLE", "Normal", np.nan, None, np.nan
        # <<< END OF MODIFIED >>>

        active_model_key_allow = selected_model_key
        model_confidence_allow = selected_model_confidence

        active_model_info = available_models.get(active_model_key_allow)
        if active_model_info and active_model_info.get('model') and active_model_info.get('features'):
            active_model_obj = active_model_info['model']
            features_for_model = active_model_info['features']
            missing_feats_ml = [f for f in features_for_model if f not in row_data.index]
            if missing_feats_ml: # pragma: no cover
                entry_allow_logger.error(f"Missing features for ML model '{active_model_key_allow}': {missing_feats_ml}")
                return False, f"ML_MISSING_FEAT_{active_model_key_allow}", "Normal", np.nan, active_model_key_allow, model_confidence_allow

            try:
                X_pred_ml = pd.DataFrame([row_data[features_for_model]], columns=features_for_model)
                global CatBoostClassifier_imported # From Part 1
                if CatBoostClassifier_imported and isinstance(active_model_obj, CatBoostClassifier_imported) and hasattr(active_model_obj, 'get_cat_feature_indices'): # pragma: no cover
                    cat_indices_pred = active_model_obj.get_cat_feature_indices()
                    for idx_cat_pred in cat_indices_pred:
                        col_name_cat_pred = X_pred_ml.columns[idx_cat_pred]
                        X_pred_ml[col_name_cat_pred] = X_pred_ml[col_name_cat_pred].astype(str)

                meta_proba_tp_allow = active_model_obj.predict_proba(X_pred_ml)[0, 1]
            except Exception as e_ml_pred: # pragma: no cover
                entry_allow_logger.error(f"Error during ML prediction with '{active_model_key_allow}': {e_ml_pred}")
                return False, f"ML_PREDICT_ERROR_{active_model_key_allow}", "Normal", np.nan, active_model_key_allow, model_confidence_allow

            if meta_proba_tp_allow < current_l1_threshold_override:
                entry_allow_logger.debug(f"Entry blocked by L1 ML Filter ({active_model_key_allow}). Proba: {meta_proba_tp_allow:.3f} < Thresh: {current_l1_threshold_override:.3f}")
                return False, f"ML_L1_FILTER_LOW_PROBA ({meta_proba_tp_allow:.3f} < {current_l1_threshold_override:.3f})", "Normal", meta_proba_tp_allow, active_model_key_allow, model_confidence_allow
        else: # pragma: no cover
            entry_allow_logger.warning(f"Selected model '{active_model_key_allow}' or its features not found in available_models. Proceeding without ML filter for this trade.")
            active_model_key_allow = "SignalOnly_Fallback" # Indicate fallback
            meta_proba_tp_allow = np.nan # No ML proba if model is missing
    else:
        meta_proba_tp_allow = np.nan # No ML filter applied

    is_reentry_trade = False
    if config.use_reentry and is_reentry_allowed(
        config=config, row_data=row_data, side=side, active_orders=active_orders,
        bars_since_last_trade=bars_since_last_trade,
        last_tp_time_for_side=last_tp_time_for_side.get(side, min_ts_val), # Get specific side's last TP time
        meta_proba_tp=meta_proba_tp_allow if pd.notna(meta_proba_tp_allow) else None
    ): # pragma: no cover
        is_reentry_trade = True
        entry_type = "Re-Entry"
    else:
        entry_type = "Normal"

    is_forced_trade = False
    if not is_reentry_trade and trade_manager.should_force_entry(
        current_time=row_data.name, signal_score=signal_score_allow,
        current_atr=row_data.get('ATR_14'), avg_atr=row_data.get('ATR_14_Rolling_Avg'),
        gain_z=row_data.get('Gain_Z'), pattern_label=str(row_data.get('Pattern_Label'))
    ): # pragma: no cover
        if hasattr(config, 'fe_ml_filter_threshold') and config.fe_ml_filter_threshold > 0:
            if pd.notna(meta_proba_tp_allow) and meta_proba_tp_allow < config.fe_ml_filter_threshold:
                entry_allow_logger.info(f"Forced Entry blocked by FE ML Filter. Proba: {meta_proba_tp_allow:.3f} < FE Thresh: {config.fe_ml_filter_threshold:.3f}")
                return False, f"FE_ML_FILTER_LOW_PROBA ({meta_proba_tp_allow:.3f} < {config.fe_ml_filter_threshold:.3f})", "Forced_Blocked", meta_proba_tp_allow, active_model_key_allow, model_confidence_allow
            elif pd.isna(meta_proba_tp_allow) and config.use_meta_classifier : # type: ignore
                entry_allow_logger.warning(f"Forced Entry blocked: ML filter is ON but Proba is NaN.")
                return False, "FE_ML_PROBA_NAN", "Forced_Blocked", meta_proba_tp_allow, active_model_key_allow, model_confidence_allow

        is_forced_trade = True
        entry_type = "Forced"
        entry_allow_logger.info(f"Forced Entry conditions met at {row_data.name} for {side}.")

    if is_reentry_trade: entry_allow_logger.debug(f"Re-Entry allowed for {side} at {row_data.name}")

    return True, "ALLOWED", entry_type, meta_proba_tp_allow, active_model_key_allow, model_confidence_allow


# --- Bar-by-Bar Exit Condition Check ---
def _check_exit_conditions_for_order(
    order: Order, config: 'StrategyConfig', bar: pd.Series, # type: ignore
    current_time: pd.Timestamp, current_bar_idx: int, logger_parent: logging.Logger # Added logger
) -> Tuple[bool, Optional[float], Optional[Dict[str, Any]]]:
    # [Patch] Added logger_parent and created specific logger
    exit_check_logger_detail = logging.getLogger(f"{logger_parent.name}._check_exit_conditions_for_order.{order.label_suffix}.{order.side}.{order.entry_time.strftime('%Y%m%d%H%M')}")
    if order.closed: return False, None, None

    bar_open = pd.to_numeric(bar.get('Open'), errors='coerce')
    bar_high = pd.to_numeric(bar.get('High'), errors='coerce')
    bar_low = pd.to_numeric(bar.get('Low'), errors='coerce')
    bar_close = pd.to_numeric(bar.get('Close'), errors='coerce')

    if any(pd.isna(p) for p in [bar_open, bar_high, bar_low, bar_close]): # pragma: no cover
        exit_check_logger_detail.warning(f"Skipping exit check at {current_time} due to NaN in OHLC.")
        return False, None, None

    exit_reason_final_check: Optional[str] = None
    exit_price_final_check: Optional[float] = None
    log_entry_for_exit_check: Optional[Dict[str, Any]] = None

    # --- [Patch] Call new TSL/BE helper functions ---
    price_for_sl_manage = bar_high if order.side == "BUY" else bar_low
    if pd.notna(price_for_sl_manage):
        # 1. Check and potentially move SL to Breakeven
        maybe_move_sl_to_be(order, config, price_for_sl_manage, current_time, exit_check_logger_detail) # type: ignore

        # 2. Update Trailing Stop Loss if active
        if order.tsl_activated:
            current_atr_for_sl = pd.to_numeric(bar.get('ATR_14'), errors='coerce')
            avg_atr_for_sl = pd.to_numeric(bar.get('ATR_14_Rolling_Avg'), errors='coerce')
            update_trailing_sl(order, config, price_for_sl_manage, current_atr_for_sl, avg_atr_for_sl, exit_check_logger_detail) # type: ignore
    else: # pragma: no cover
        exit_check_logger_detail.warning(f"price_for_sl_manage is NaN at {current_time}. SL not updated for BE/TSL.")
    # --- END OF [Patch] ---

    # Check SL Hit (after potential TSL/BE adjustments)
    if order.side == "BUY" and bar_low <= order.sl_price:
        exit_price_final_check = order.sl_price
        exit_reason_final_check = "BE-SL" if order.be_triggered and math.isclose(order.sl_price, order.entry_price) else "SL"
        exit_check_logger_detail.debug(f"BUY SL HIT: Low={bar_low:.5f} <= SL={order.sl_price:.5f}. Reason={exit_reason_final_check}")
    elif order.side == "SELL" and bar_high >= order.sl_price:
        exit_price_final_check = order.sl_price
        exit_reason_final_check = "BE-SL" if order.be_triggered and math.isclose(order.sl_price, order.entry_price) else "SL"
        exit_check_logger_detail.debug(f"SELL SL HIT: High={bar_high:.5f} >= SL={order.sl_price:.5f}. Reason={exit_reason_final_check}")

    # Check TP Hit (only if SL not hit in the same bar logic pass)
    if exit_reason_final_check is None:
        if order.side == "BUY" and bar_high >= order.tp_price: # tp_price is TP2
            exit_price_final_check = order.tp_price
            exit_reason_final_check = "TP"
            exit_check_logger_detail.debug(f"BUY TP2 HIT: High={bar_high:.5f} >= TP2={order.tp_price:.5f}")
        elif order.side == "SELL" and bar_low <= order.tp_price: # tp_price is TP2
            exit_price_final_check = order.tp_price
            exit_reason_final_check = "TP"
            exit_check_logger_detail.debug(f"SELL TP2 HIT: Low={bar_low:.5f} <= TP2={order.tp_price:.5f}")

    # Check Max Holding Bars (if no SL/TP hit yet)
    if exit_reason_final_check is None and \
       order.config_at_entry.max_holding_bars is not None and \
       order.holding_bars >= order.config_at_entry.max_holding_bars: # pragma: no cover
        exit_price_final_check = bar_close # Exit at close of the max holding bar
        exit_reason_final_check = f"MaxBars ({order.config_at_entry.max_holding_bars})"
        exit_check_logger_detail.debug(f"MAX HOLDING BARS: Holding={order.holding_bars}, Max={order.config_at_entry.max_holding_bars}")

    if exit_reason_final_check and pd.notna(exit_price_final_check):
        order.closed = True # Mark order as closed internally
        order.exit_reason_internal = exit_reason_final_check # Store internal reason
        # Prepare log entry data, to be finalized by close_trade
        log_entry_for_exit_check = order.to_dict()
        log_entry_for_exit_check.update({
            "close_time": current_time,
            "exit_price": exit_price_final_check,
            "exit_reason": exit_reason_final_check, # This will be the primary reason
            "lot_closed_this_event": order.lot, # Assume full close for now, PTP handles partials
            "remaining_lot_after_event": 0.0,
            "is_partial_tp_event": False, # This is for full exits
            "current_partial_tp_level_processed": len(order.partial_tp_processed_levels),
        })
        exit_check_logger_detail.info(f"Exit Finalized: Reason: {exit_reason_final_check}, Price: {exit_price_final_check:.5f}")
        return True, exit_price_final_check, log_entry_for_exit_check

    return False, None, None

# --- Close Trade Function ---
def close_trade(
    order: Order, config: 'StrategyConfig', exit_price: float, exit_time: pd.Timestamp, # type: ignore
    exit_reason: str, lot_closed: float,
    trade_log_list_ct: List[Dict[str, Any]], equity_tracker_dict_ct: Dict[str, Any],
    run_summary_dict_ct: Optional[Dict[str, Any]], label_str_ct: str
):
    close_trade_logger_detail = logging.getLogger(f"{__name__}.close_trade.{order.label_suffix}.{order.side}.{order.entry_time.strftime('%Y%m%d%H%M')}")
    close_trade_logger_detail.info(f"   Closing Trade/Portion: Reason: {exit_reason}, Lot Closed: {lot_closed:.2f}, Exit Price: {exit_price:.5f if pd.notna(exit_price) else 'N/A'}")

    pnl_points_ct_val = 0.0; pnl_points_net_spread_ct_val = 0.0; raw_pnl_usd_ct_val = 0.0
    commission_usd_ct_val = 0.0; spread_cost_usd_ct_val = 0.0; slippage_usd_ct_val = 0.0; net_pnl_usd_ct_val = 0.0

    entry_price_num_ct_val = pd.to_numeric(order.entry_price, errors='coerce')
    exit_price_num_ct_val = pd.to_numeric(exit_price, errors='coerce')
    lot_size_num_closed_val = pd.to_numeric(lot_closed, errors='coerce')
    equity_before_this_action_ct_val = equity_tracker_dict_ct['current_equity']

    if pd.notna(entry_price_num_ct_val) and pd.notna(exit_price_num_ct_val) and pd.notna(lot_size_num_closed_val) and lot_size_num_closed_val >= config.min_lot:
        is_be_sl_full_close_ct = "BE-SL" in exit_reason.upper() and not ("Partial TP" in exit_reason or "PTP" in exit_reason)

        if is_be_sl_full_close_ct:
            # For BE-SL, PnL points are 0. Costs are spread and commission. No slippage.
            pnl_points_ct_val = 0.0
            spread_cost_usd_ct_val = config.spread_points * (lot_size_num_closed_val / config.min_lot) * config.point_value
            commission_usd_ct_val = (lot_size_num_closed_val / config.min_lot) * config.commission_per_001_lot
            slippage_usd_ct_val = 0.0 # No slippage on BE-SL
            net_pnl_usd_ct_val = 0 - spread_cost_usd_ct_val - commission_usd_ct_val
            pnl_points_net_spread_ct_val = pnl_points_ct_val - config.spread_points # Still calculate for consistency
            raw_pnl_usd_ct_val = pnl_points_net_spread_ct_val * (lot_size_num_closed_val / config.min_lot) * config.point_value
        else:
            # Standard PnL calculation
            if order.side == "BUY":
                pnl_points_ct_val = (exit_price_num_ct_val - entry_price_num_ct_val) * 10.0
            else: # SELL
                pnl_points_ct_val = (entry_price_num_ct_val - exit_price_num_ct_val) * 10.0

            pnl_points_net_spread_ct_val = pnl_points_ct_val - config.spread_points
            spread_cost_usd_ct_val = config.spread_points * (lot_size_num_closed_val / config.min_lot) * config.point_value
            raw_pnl_usd_ct_val = pnl_points_net_spread_ct_val * (lot_size_num_closed_val / config.min_lot) * config.point_value
            commission_usd_ct_val = (lot_size_num_closed_val / config.min_lot) * config.commission_per_001_lot

            # Apply slippage only if not a system/max_bars/BE/PTP exit
            min_slip_cfg_ct_val = getattr(config, 'min_slippage_points', -5.0)
            max_slip_cfg_ct_val = getattr(config, 'max_slippage_points', -1.0)
            apply_slippage_ct = not ("MaxBars" in exit_reason or "BE-SL" in exit_reason.upper() or "PTP" in exit_reason or "Partial TP" in exit_reason or "KillSwitch" in exit_reason or "MARGIN_CALL" in exit_reason)
            if apply_slippage_ct: # pragma: no cover
                slippage_points_local_ct_val = random.uniform(min_slip_cfg_ct_val, max_slip_cfg_ct_val)
                slippage_usd_ct_val = slippage_points_local_ct_val * (lot_size_num_closed_val / config.min_lot) * config.point_value
            else:
                slippage_usd_ct_val = 0.0

            net_pnl_usd_ct_val = raw_pnl_usd_ct_val - commission_usd_ct_val + slippage_usd_ct_val
    else: # pragma: no cover
        # Handle cases where PnL cannot be calculated (e.g., invalid price/lot)
        # Still apply costs if lot_closed is valid
        close_trade_logger_detail.warning(f"Order {order.entry_time}: PnL for this action set to 0. Invalid price/lot. Reason: {exit_reason}, Lot Closed: {lot_closed}")
        if pd.notna(lot_size_num_closed_val) and lot_size_num_closed_val >= config.min_lot:
            spread_cost_usd_ct_val = config.spread_points * (lot_size_num_closed_val / config.min_lot) * config.point_value
            commission_usd_ct_val = (lot_size_num_closed_val / config.min_lot) * config.commission_per_001_lot
            net_pnl_usd_ct_val = 0 - spread_cost_usd_ct_val - commission_usd_ct_val # Net loss is costs

    # Update equity
    equity_tracker_dict_ct['current_equity'] += net_pnl_usd_ct_val
    equity_tracker_dict_ct['peak_equity'] = max(equity_tracker_dict_ct['peak_equity'], equity_tracker_dict_ct['current_equity'])
    if 'history' in equity_tracker_dict_ct and isinstance(equity_tracker_dict_ct['history'], dict):
        equity_tracker_dict_ct['history'][exit_time] = equity_tracker_dict_ct['current_equity']

    # Update run summary
    if run_summary_dict_ct and isinstance(run_summary_dict_ct, dict): # pragma: no cover
        run_summary_dict_ct['total_commission'] = run_summary_dict_ct.get('total_commission', 0.0) + commission_usd_ct_val
        run_summary_dict_ct['total_spread'] = run_summary_dict_ct.get('total_spread', 0.0) + spread_cost_usd_ct_val
        run_summary_dict_ct['total_slippage'] = run_summary_dict_ct.get('total_slippage', 0.0) + abs(slippage_usd_ct_val) # Slippage is usually negative
        if is_be_sl_full_close_ct:
            run_summary_dict_ct['be_sl_triggered_count'] = run_summary_dict_ct.get('be_sl_triggered_count', 0) + 1
        if "TSL" in exit_reason.upper() or ("SL" in exit_reason.upper() and order.tsl_activated and not order.be_triggered): # Count TSL hits that are not BE
            run_summary_dict_ct['tsl_triggered_count'] = run_summary_dict_ct.get('tsl_triggered_count', 0) + 1


    # Log trade details
    log_entry_data_final_ct = order.to_dict() # Get current state of order
    log_entry_data_final_ct.update({
        "period": label_str_ct,
        "close_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "lot_closed_this_event": lot_closed,
        "remaining_lot_after_event": round(order.lot - lot_closed, 2) if ("Partial TP" in exit_reason or "PTP" in exit_reason) and not order.closed else 0.0,
        "final_sl_price": order.sl_price, # SL price at the moment of this event
        "pnl_points_gross": pnl_points_ct_val,
        "pnl_points_net_spread": pnl_points_net_spread_ct_val,
        "pnl_usd_gross": raw_pnl_usd_ct_val,
        "commission_usd": commission_usd_ct_val,
        "spread_cost_usd": spread_cost_usd_ct_val,
        "slippage_usd": slippage_usd_ct_val,
        "pnl_usd_net": net_pnl_usd_ct_val,
        "equity_before_this_action": equity_before_this_action_ct_val,
        "equity_after_this_action": equity_tracker_dict_ct['current_equity'],
        "is_partial_tp_event": "Partial TP" in exit_reason or "PTP" in exit_reason,
        "current_partial_tp_level_processed": len(order.partial_tp_processed_levels) + (1 if ("Partial TP" in exit_reason or "PTP" in exit_reason) and not order.closed else 0),
    })
    if isinstance(trade_log_list_ct, list):
        trade_log_list_ct.append(log_entry_data_final_ct)
    else: # pragma: no cover
        close_trade_logger_detail.error("trade_log_list_ct is not a list. Cannot append entry.")

    close_trade_logger_detail.debug(f"Trade event logged: {exit_reason}, PnL Net: {net_pnl_usd_ct_val:.2f}")


# --- Backtesting Simulation Engine (Main Function - Full Logic) ---
# <<< MODIFIED: run_backtest_simulation_v34 correctly processes potentially None model_key. >>>
# <<< MODIFIED: [Patch] run_backtest_simulation_v34 now calls new TSL/BE helpers and _check_kill_switch. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.1] Integrated all new helper functions into run_backtest_simulation_v34 and refined logic. >>>
def run_backtest_simulation_v34(
    df_m1_segment_pd: pd.DataFrame, label: str, initial_capital_segment: float, side: str = "BUY",
    config_obj: Optional['StrategyConfig'] = None, risk_manager_obj: Optional['RiskManager'] = None, trade_manager_obj: Optional['TradeManager'] = None,  # type: ignore
    fund_profile: Optional[Dict[str, Any]] = None, fold_config_override: Optional[Dict[str, Any]] = None,
    available_models: Optional[Dict[str, Any]] = None,
    model_switcher_func: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[Optional[str], Optional[float]]]] = None, # <<< MODIFIED: Type Hint
    meta_min_proba_thresh_override: Optional[float] = None, current_fold_index: Optional[int] = None,
    initial_kill_switch_state: bool = False, initial_consecutive_losses: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[pd.Timestamp, float], float, Dict[str, Any], List[Dict[str, Any]], Optional[str], Optional[str], bool, int, float]:

    sim_logger = logging.getLogger(f"{__name__}.run_backtest_simulation_v34.{label}.{side}")
    sim_logger.info(f"--- Starting Backtest Simulation (v4.9.23 - AI Studio Patch v4.9.1): Label='{label}', Side='{side}', Fold={current_fold_index if current_fold_index is not None else 'N/A'} ---")

    if config_obj is None: # pragma: no cover
        sim_logger.warning(f"[{label}] StrategyConfig (config_obj) not provided. Initializing with default empty config.")
        config_obj = StrategyConfig({})  # type: ignore
    if risk_manager_obj is None: # pragma: no cover
        sim_logger.warning(f"[{label}] RiskManager (risk_manager_obj) not provided. Initializing new one.")
        risk_manager_obj = RiskManager(config_obj)  # type: ignore
        risk_manager_obj.dd_peak = initial_capital_segment # Initialize peak if RM is new
    if trade_manager_obj is None: # pragma: no cover
        sim_logger.warning(f"[{label}] TradeManager (trade_manager_obj) not provided. Initializing new one.")
        trade_manager_obj = TradeManager(config_obj, risk_manager_obj)  # type: ignore

    # Use provided objects or initialize if None
    current_fund_profile = fund_profile if fund_profile else {"name": getattr(config_obj, 'default_fund_name', "DEFAULT"), "risk": config_obj.risk_per_trade, "mm_mode": "balanced"}
    current_fold_config = fold_config_override if fold_config_override else {}
    l1_threshold_run = meta_min_proba_thresh_override if meta_min_proba_thresh_override is not None else config_obj.meta_min_proba_thresh
    timeframe_m1_for_sim = config_obj.timeframe_minutes_m1

    equity_tracker: Dict[str, Any] = {
        'current_equity': initial_capital_segment,
        'peak_equity': risk_manager_obj.dd_peak if risk_manager_obj.dd_peak is not None and risk_manager_obj.dd_peak >= initial_capital_segment else initial_capital_segment,
        'history': {df_m1_segment_pd.index[0] if not df_m1_segment_pd.empty else pd.Timestamp.now(tz='UTC'): initial_capital_segment}
    }
    if risk_manager_obj.dd_peak is None or risk_manager_obj.dd_peak < initial_capital_segment: # Ensure peak is at least initial capital
        risk_manager_obj.dd_peak = initial_capital_segment

    active_orders: List[Order] = []
    trade_log_buffer: List[Dict[str, Any]] = []
    blocked_order_log: List[Dict[str, Any]] = []
    run_summary: Dict[str, Any] = {
        "total_commission": 0.0, "total_spread": 0.0, "total_slippage": 0.0,
        "be_sl_triggered_count": 0, "tsl_triggered_count": 0,
        "fund_profile": current_fund_profile, "total_ib_lot_accumulator": 0.0,
        "orders_blocked_dd": 0, "orders_blocked_cooldown": 0, "orders_skipped_ml_l1": 0,
        "orders_blocked_no_model": 0 # For tracking when model_switcher returns (None, None)
    }
    max_drawdown_pct_overall = 0.0
    bars_since_last_trade = 0 # For re-entry cooldown
    kill_switch_activated_runtime = initial_kill_switch_state # Use passed-in state
    consecutive_losses_runtime = initial_consecutive_losses # Use passed-in state
    last_n_full_trade_pnls: List[float] = [] # For soft kill cooldown
    current_risk_mode = "normal"
    trade_history_reasons: List[str] = [] # For TP2 lot boost
    error_in_loop_runtime = False
    min_ts_for_sim = pd.Timestamp("2000-01-01", tz="UTC") if not df_m1_segment_pd.empty and df_m1_segment_pd.index.tz is not None else pd.Timestamp("2000-01-01")
    last_tp_time_for_side_dict: Dict[str, pd.Timestamp] = defaultdict(lambda: min_ts_for_sim)


    label_suffix_df = f"_{label}"
    df_sim = df_m1_segment_pd.copy()
    df_sim = _predefine_result_columns_for_simulation(df_sim, label_suffix_df)

    sim_logger.info(f"[{label}] Initializing simulation. Capital: ${initial_capital_segment:.2f}, Risk: {current_fund_profile.get('risk', config_obj.risk_per_trade):.3%}, MM Mode: {current_fund_profile.get('mm_mode', 'N/A')}")
    sim_logger.info(f"[{label}] Config: MaxHold={config_obj.max_holding_bars}, PTP En={config_obj.enable_partial_tp}, ReEn={config_obj.use_reentry}, FE En={config_obj.enable_forced_entry}")
    sim_logger.info(f"[{label}] Initial KillSwitchState={kill_switch_activated_runtime}, InitialConsecLosses={consecutive_losses_runtime}")


    sim_iterator = tqdm(df_sim.iterrows(), total=df_sim.shape[0], desc=f"  Sim ({label}, {side})", leave=False, mininterval=2.0) if tqdm else df_sim.iterrows()
    current_bar_idx = 0 # For should_exit_due_to_holding

    meta_model_type_used_runtime: Optional[str] = "N/A"
    meta_meta_model_type_used_runtime: Optional[str] = "N/A"

    try:
        for idx_bar, row_data_bar in sim_iterator:
            now_bar = idx_bar # This is the timestamp of the current bar
            equity_at_bar_start = equity_tracker['current_equity']
            equity_change_this_bar = 0.0 # Accumulates PnL from trades closed this bar
            order_opened_this_bar = False
            sim_logger.debug(f"--- Bar {current_bar_idx} ({now_bar}) --- Equity: {equity_at_bar_start:.2f}, Active: {len(active_orders)}, ConsLoss: {consecutive_losses_runtime}, KS_Active: {kill_switch_activated_runtime} ---")

            # --- [Patch] Kill Switch Checks (Primary Logic) ---
            # 1. Margin Call (most critical)
            if check_margin_call(equity_tracker['current_equity']): # type: ignore # pragma: no cover
                kill_switch_activated_runtime = True
                risk_manager_obj.soft_kill_active = True # Also activate soft kill to prevent new entries
                sim_logger.critical(f"MARGIN CALL at {now_bar}. Equity depleted. Forcing Kill Switch.")
                for mc_order in active_orders:
                    if not mc_order.closed:
                        close_price_mc_val = pd.to_numeric(row_data_bar.get("Close", mc_order.entry_price), errors='coerce')
                        if pd.isna(close_price_mc_val): close_price_mc_val = mc_order.entry_price
                        close_trade(mc_order, config_obj, close_price_mc_val, now_bar, "MARGIN_CALL", mc_order.lot, trade_log_buffer, equity_tracker, run_summary, label)
                        mc_order.closed_by_killswitch = True # Mark as closed by system
                active_orders.clear()
                break # Exit simulation loop immediately

            # 2. RiskManager's internal DD check (can raise RuntimeError)
            try:
                current_dd_rm_val = risk_manager_obj.update_drawdown(equity_tracker['current_equity'])
                max_drawdown_pct_overall = max(max_drawdown_pct_overall, current_dd_rm_val)
            except RuntimeError as e_kill_dd_loop: # Raised by RiskManager if hard DD threshold hit # pragma: no cover
                sim_logger.critical(f"KILL SWITCH (Drawdown from RiskManager) Triggered: {e_kill_dd_loop} at {now_bar}")
                kill_switch_activated_runtime = True # Ensure our flag is set

            # 3. Use the new _check_kill_switch to consolidate and log
            # This function now handles both DD and consecutive loss checks.
            _, kill_switch_activated_runtime = _check_kill_switch( # type: ignore
                current_equity=equity_tracker['current_equity'],
                peak_equity=equity_tracker['peak_equity'],
                kill_switch_dd_threshold_config=config_obj.kill_switch_dd,
                kill_switch_consecutive_losses_config=config_obj.kill_switch_consecutive_losses,
                current_consecutive_losses=consecutive_losses_runtime,
                kill_switch_active_status=kill_switch_activated_runtime, # Pass current status
                current_time=now_bar,
                config=config_obj,
                logger_parent=sim_logger
            )
            # --- END OF [Patch] ---

            if kill_switch_activated_runtime: # pragma: no cover
                sim_logger.warning(f"Kill switch active at {now_bar}. Closing any remaining orders & stopping new entries.")
                for order_ks_loop_close in active_orders:
                    if not order_ks_loop_close.closed:
                        exit_price_ks_val = pd.to_numeric(row_data_bar.get("Close"), errors='coerce')
                        if pd.isna(exit_price_ks_val): exit_price_ks_val = order_ks_loop_close.entry_price
                        close_trade(order_ks_loop_close, config_obj, exit_price_ks_val, now_bar, "KillSwitchActivated", order_ks_loop_close.lot, trade_log_buffer, equity_tracker, run_summary, label)
                        order_ks_loop_close.closed_by_killswitch = True
                active_orders.clear()
                break # Exit simulation loop

            # --- Process Active Orders (Exits, PTP, TSL, BE) ---
            next_active_orders_loop: List[Order] = []
            for order_item in active_orders:
                if order_item.closed: continue # Skip already closed portions
                order_item.holding_bars += 1

                # Partial Take Profit (PTP)
                if config_obj.enable_partial_tp and order_item.lot >= config_obj.min_lot and pd.notna(order_item.atr_at_entry) and order_item.atr_at_entry > 1e-9:
                    sl_multiplier_for_ptp = current_fold_config.get('sl_multiplier', getattr(order_item.config_at_entry, 'default_sl_multiplier', 1.5))
                    sl_delta_for_ptp_item = order_item.atr_at_entry * sl_multiplier_for_ptp
                    if sl_delta_for_ptp_item > 1e-9:
                        for ptp_idx_item, ptp_level_item in enumerate(config_obj.partial_tp_levels):
                            if ptp_idx_item not in order_item.partial_tp_processed_levels:
                                ptp_r_item = ptp_level_item["r_multiple"]
                                ptp_price_target_item = order_item.entry_price + (sl_delta_for_ptp_item * ptp_r_item) if order_item.side == "BUY" else order_item.entry_price - (sl_delta_for_ptp_item * ptp_r_item)
                                ptp_hit_item = False; ptp_exit_price_item = np.nan
                                bar_h_ptp_val = pd.to_numeric(row_data_bar.get('High'), errors='coerce'); bar_l_ptp_val = pd.to_numeric(row_data_bar.get('Low'), errors='coerce')
                                if order_item.side == "BUY" and pd.notna(bar_h_ptp_val) and bar_h_ptp_val >= ptp_price_target_item: ptp_hit_item = True; ptp_exit_price_item = ptp_price_target_item
                                elif order_item.side == "SELL" and pd.notna(bar_l_ptp_val) and bar_l_ptp_val <= ptp_price_target_item: ptp_hit_item = True; ptp_exit_price_item = ptp_price_target_item
                                if ptp_hit_item:
                                    sim_logger.info(f"   PTP Level {ptp_idx_item + 1} ({ptp_r_item:.1f}R) hit for order {order_item.entry_idx} at {now_bar}.")
                                    lot_to_close_ptp_val_item = round(min(order_item.original_lot * ptp_level_item["close_pct"], order_item.lot), 2); lot_to_close_ptp_val_item = max(lot_to_close_ptp_val_item, config_obj.min_lot)
                                    remaining_lot_after_ptp_val = round(order_item.lot - lot_to_close_ptp_val_item, 2); ptp_exit_reason_str_item = f"Partial TP {ptp_idx_item + 1} ({ptp_r_item:.1f}R)"
                                    if 0 < remaining_lot_after_ptp_val < config_obj.min_lot: lot_to_close_ptp_val_item = order_item.lot; remaining_lot_after_ptp_val = 0.0; order_item.closed = True; ptp_exit_reason_str_item = f"Full Close on PTP {ptp_idx_item + 1}"
                                    elif remaining_lot_after_ptp_val <= 1e-9: lot_to_close_ptp_val_item = order_item.lot; remaining_lot_after_ptp_val = 0.0; order_item.closed = True; ptp_exit_reason_str_item = f"Full Close on PTP {ptp_idx_item + 1} (Final)"
                                    if lot_to_close_ptp_val_item >= config_obj.min_lot:
                                        equity_before_this_ptp_action_val = equity_tracker['current_equity']
                                        close_trade(order_item, config_obj, ptp_exit_price_item, now_bar, ptp_exit_reason_str_item, lot_to_close_ptp_val_item, trade_log_buffer, equity_tracker, run_summary, label)
                                        equity_change_this_bar += (equity_tracker['current_equity'] - equity_before_this_ptp_action_val)
                                        order_item.lot = remaining_lot_after_ptp_val; order_item.partial_tp_processed_levels.add(ptp_idx_item); order_item.reached_tp1 = True
                                        if config_obj.partial_tp_move_sl_to_entry and not order_item.be_triggered:
                                            if not math.isclose(order_item.sl_price, order_item.entry_price):
                                                sim_logger.info(f"      [PTP-BE] Moving SL to Entry ({order_item.entry_price:.5f}) after PTP for order {order_item.entry_idx}.")
                                                order_item.sl_price = order_item.entry_price; order_item.be_triggered = True; order_item.be_triggered_time = now_bar
                                                if run_summary and 'be_sl_triggered_count' in run_summary: run_summary['be_sl_triggered_count'] +=1 # Count BE from PTP
                                                if order_item.tsl_activated: order_item.trailing_sl_price = order_item.entry_price # Adjust TSL base if active
                                    if order_item.closed: break # Break from PTP levels loop if order fully closed
                            if order_item.closed: break # Break from PTP check if order fully closed

                # TSL Activation (before BE check for this bar, as TSL might activate based on current bar's high/low before BE condition is met)
                if not order_item.closed and not order_item.tsl_activated and pd.notna(order_item.atr_at_entry) and order_item.atr_at_entry > 1e-9:
                    tsl_activation_price_diff_val_act = order_item.atr_at_entry * config_obj.adaptive_tsl_start_atr_mult
                    bar_h_tsl_act_val_curr = pd.to_numeric(row_data_bar.get('High'), errors='coerce'); bar_l_tsl_act_val_curr = pd.to_numeric(row_data_bar.get('Low'), errors='coerce')
                    if order_item.side == "BUY" and pd.notna(bar_h_tsl_act_val_curr) and bar_h_tsl_act_val_curr >= order_item.entry_price + tsl_activation_price_diff_val_act:
                        order_item.tsl_activated = True
                        order_item.peak_since_tsl_activation = bar_h_tsl_act_val_curr # Initialize peak
                        sim_logger.info(f"   TSL Activated for BUY order {order_item.entry_idx} at price {bar_h_tsl_act_val_curr:.5f}")
                    elif order_item.side == "SELL" and pd.notna(bar_l_tsl_act_val_curr) and bar_l_tsl_act_val_curr <= order_item.entry_price - tsl_activation_price_diff_val_act:
                        order_item.tsl_activated = True
                        order_item.trough_since_tsl_activation = bar_l_tsl_act_val_curr # Initialize trough
                        sim_logger.info(f"   TSL Activated for SELL order {order_item.entry_idx} at price {bar_l_tsl_act_val_curr:.5f}")

                # Main Exit Conditions (SL, TP, MaxBars) - This now includes TSL/BE logic via _check_exit_conditions_for_order
                if not order_item.closed:
                    is_exited_main_val_call, exit_price_main_val_item_call, log_entry_main_val_item_call = _check_exit_conditions_for_order(
                        order_item, config_obj, row_data_bar, now_bar, current_bar_idx, sim_logger # Pass parent logger
                    )
                    if is_exited_main_val_call and log_entry_main_val_item_call and pd.notna(exit_price_main_val_item_call):
                        equity_before_this_final_close_call = equity_tracker['current_equity']
                        close_trade(order_item, config_obj, exit_price_main_val_item_call, now_bar, log_entry_main_val_item_call['exit_reason'], order_item.lot, trade_log_buffer, equity_tracker, run_summary, label)
                        equity_change_this_bar += (equity_tracker['current_equity'] - equity_before_this_final_close_call)
                        finalized_log_entry = trade_log_buffer[-1]; trade_history_reasons.append(finalized_log_entry["exit_reason"])
                        if finalized_log_entry.get("pnl_usd_net", 0.0) < 0:
                            consecutive_losses_runtime += 1
                            if order_item.is_forced_entry: trade_manager_obj.update_forced_entry_result(is_loss=True)
                        else:
                            consecutive_losses_runtime = 0 # Reset on any win or BE
                            if finalized_log_entry["exit_reason"].upper() == "TP": # Only update last TP time on full TP
                                last_tp_time_for_side_dict[order_item.side] = now_bar
                            if order_item.is_forced_entry: trade_manager_obj.update_forced_entry_result(is_loss=False)
                        last_n_full_trade_pnls.append(finalized_log_entry.get("pnl_usd_net", 0.0))
                        if len(last_n_full_trade_pnls) > config_obj.soft_cooldown_lookback: last_n_full_trade_pnls.pop(0) # Maintain lookback window
                    else: # If not exited by SL/TP/MaxBars, it's still active
                        next_active_orders_loop.append(order_item)
            active_orders = [o for o in next_active_orders_loop if not o.closed] # Update active_orders list

            # --- New Order Entry Logic ---
            if not kill_switch_activated_runtime: # Only consider new entries if KS not active
                can_open_new_val_call, block_reason_open_val_call, entry_type_open_val_call, meta_proba_open_val_call, model_key_open_val_call, model_conf_open_val_call = is_entry_allowed(
                    config=config_obj, risk_manager=risk_manager_obj, trade_manager=trade_manager_obj,
                    row_data=row_data_bar, session_tag=str(row_data_bar.get("session", "Other")),
                    consecutive_losses_run=consecutive_losses_runtime,
                    equity=(equity_at_bar_start + equity_change_this_bar), # Use updated equity for lot sizing
                    peak_equity=equity_tracker['peak_equity'],
                    initial_capital_segment=initial_capital_segment, side=side, active_orders=active_orders,
                    bars_since_last_trade=bars_since_last_trade, last_tp_time_for_side=last_tp_time_for_side_dict,
                    fold_config_override=current_fold_config, available_models=available_models,
                    model_switcher_func=model_switcher_func,
                    current_l1_threshold_override=l1_threshold_run,
                    min_ts_val=min_ts_for_sim, label_suffix_entry=label_suffix_df
                )

                # <<< MODIFIED: Process potentially None model_key >>>
                if model_key_open_val_call is not None:
                    meta_model_type_used_runtime = model_key_open_val_call
                elif block_reason_open_val_call == "NO_VALID_MODEL_AVAILABLE": # Check specific reason # pragma: no cover
                    meta_model_type_used_runtime = "NO_MODEL_SELECTED" # Log this state
                    if run_summary: run_summary["orders_blocked_no_model"] = run_summary.get("orders_blocked_no_model", 0) + 1
                # <<< END OF MODIFIED >>>

                if can_open_new_val_call:
                    order_opened_this_bar = True
                    trade_manager_obj.update_last_trade_time(now_bar)
                    atr_at_entry_val_open_call = pd.to_numeric(row_data_bar.get("ATR_14_Shifted"), errors='coerce')
                    if pd.isna(atr_at_entry_val_open_call) or atr_at_entry_val_open_call < 1e-9: # pragma: no cover
                        sim_logger.warning(f"   Cannot open order at {now_bar}: Invalid ATR_Shifted ({atr_at_entry_val_open_call}).")
                    else:
                        entry_price_new_val_call = pd.to_numeric(row_data_bar.get("Open"), errors='coerce')
                        if pd.isna(entry_price_new_val_call): sim_logger.error(f"  Cannot open order at {now_bar}: Invalid Open price."); continue # pragma: no cover

                        sl_multiplier_open_eff_call = current_fold_config.get('sl_multiplier', config_obj.default_sl_multiplier)
                        sl_delta_open_val_calc_call = atr_at_entry_val_open_call * sl_multiplier_open_eff_call
                        original_sl_open_val_calc_call = entry_price_new_val_call - sl_delta_open_val_calc_call if side == "BUY" else entry_price_new_val_call + sl_delta_open_val_calc_call

                        tp1_r_mult_open_val_calc_call = config_obj.partial_tp_levels[0]['r_multiple'] if config_obj.enable_partial_tp and config_obj.partial_tp_levels else 1.0
                        tp1_price_open_val_calc_val_call = entry_price_new_val_call + (sl_delta_open_val_calc_call * tp1_r_mult_open_val_calc_call) if side == "BUY" else entry_price_new_val_call - (sl_delta_open_val_calc_call * tp1_r_mult_open_val_calc_call)

                        current_atr_open_val_calc_call = pd.to_numeric(row_data_bar.get("ATR_14"), errors='coerce')
                        avg_atr_open_val_calc_val_call = pd.to_numeric(row_data_bar.get("ATR_14_Rolling_Avg"), errors='coerce')
                        tp2_r_mult_open_val_calc_call = dynamic_tp2_multiplier(config_obj, current_atr_open_val_calc_call, avg_atr_open_val_calc_val_call)
                        tp2_price_open_val_calc_final_call = entry_price_new_val_call + (sl_delta_open_val_calc_call * tp2_r_mult_open_val_calc_call) if side == "BUY" else entry_price_new_val_call - (sl_delta_open_val_calc_call * tp2_r_mult_open_val_calc_call)

                        equity_for_lot_open_val_call = equity_at_bar_start + equity_change_this_bar
                        mm_mode_open_val_call = current_fund_profile.get('mm_mode', 'balanced')
                        risk_pct_open_val_call = current_fund_profile.get('risk', config_obj.risk_per_trade)
                        base_lot_new_val_call = calculate_lot_by_fund_mode(config_obj, mm_mode_open_val_call, risk_pct_open_val_call, equity_for_lot_open_val_call, atr_at_entry_val_open_call, sl_delta_open_val_calc_call)
                        boosted_lot_new_val_call = adjust_lot_tp2_boost(config_obj, trade_history_reasons, base_lot_new_val_call)
                        final_lot_new_val_call, risk_mode_new_val_call = adjust_lot_recovery_mode(config_obj, boosted_lot_new_val_call, consecutive_losses_runtime) # type: ignore

                        if final_lot_new_val_call >= config_obj.min_lot:
                            if run_summary and isinstance(run_summary, dict): run_summary['total_ib_lot_accumulator'] = run_summary.get('total_ib_lot_accumulator', 0.0) + final_lot_new_val_call
                            enable_ttp2_new_val_call = pd.notna(current_atr_open_val_calc_call) and current_atr_open_val_calc_call > config_obj.ttp2_atr_threshold_activate

                            new_order_instance_val_call = Order(
                                entry_idx=idx_bar, entry_time=now_bar, entry_price=entry_price_new_val_call,
                                original_lot=final_lot_new_val_call, lot_size=final_lot_new_val_call,
                                original_sl_price=original_sl_open_val_calc_call, sl_price=original_sl_open_val_calc_call,
                                tp_price=tp2_price_open_val_calc_final_call, tp1_price=tp1_price_open_val_calc_val_call,
                                entry_bar_count=current_bar_idx, side=side,
                                m15_trend_zone=str(row_data_bar.get("Trend_Zone", "NEUTRAL")), trade_tag=str(row_data_bar.get("Trade_Tag", "N/A")),
                                signal_score=row_data_bar.get('Signal_Score', 0.0),
                                trade_reason=str(row_data_bar.get("Trade_Reason", "NONE")) if entry_type_open_val_call != "Forced" else f"FORCED_{str(row_data_bar.get('Trade_Reason', 'NONE'))}",
                                session=str(row_data_bar.get("session", "Other")), pattern_label_entry=str(row_data_bar.get("Pattern_Label", "Normal")),
                                is_reentry=(entry_type_open_val_call == "Re-Entry"), is_forced_entry=(entry_type_open_val_call == "Forced"),
                                meta_proba_tp=meta_proba_open_val_call, meta2_proba_tp=np.nan,
                                atr_at_entry=atr_at_entry_val_open_call, equity_before_open=equity_for_lot_open_val_call,
                                entry_gain_z=pd.to_numeric(row_data_bar.get("Gain_Z"), errors='coerce'),
                                entry_macd_smooth=pd.to_numeric(row_data_bar.get("MACD_hist_smooth"), errors='coerce'),
                                entry_candle_ratio=pd.to_numeric(row_data_bar.get("Candle_Ratio"), errors='coerce'),
                                entry_adx=pd.to_numeric(row_data_bar.get("ADX"), errors='coerce'),
                                entry_volatility_index=pd.to_numeric(row_data_bar.get("Volatility_Index"), errors='coerce'),
                                risk_mode_at_entry=risk_mode_new_val_call,
                                use_trailing_for_tp2=enable_ttp2_new_val_call,
                                trailing_start_price=tp1_price_open_val_calc_val_call if enable_ttp2_new_val_call else np.nan, # Activate TSL for TP2 when TP1 is hit
                                trailing_step_r=config_obj.adaptive_tsl_default_step_r if enable_ttp2_new_val_call else np.nan,
                                active_model_at_entry=model_key_open_val_call, model_confidence_at_entry=model_conf_open_val_call,
                                label_suffix=label, config_at_entry=config_obj
                            )
                            active_orders.append(new_order_instance_val_call)
                            sim_logger.info(f"   +++ ORDER OPENED ({entry_type_open_val_call}): {side} Lot={final_lot_new_val_call:.2f} @{entry_price_new_val_call:.5f}, SL={original_sl_open_val_calc_call:.5f}, TP1={tp1_price_open_val_calc_val_call:.5f}, TP2={tp2_price_open_val_calc_final_call:.5f} Model: {model_key_open_val_call} Conf: {model_conf_open_val_call} +++")

                            # Log order details to df_sim
                            df_sim.loc[idx_bar, f"Order_Opened{label_suffix_df}"] = True
                            df_sim.loc[idx_bar, f"Lot_Size{label_suffix_df}"] = final_lot_new_val_call
                            df_sim.loc[idx_bar, f"Entry_Price_Actual{label_suffix_df}"] = entry_price_new_val_call
                            df_sim.loc[idx_bar, f"SL_Price_Actual{label_suffix_df}"] = original_sl_open_val_calc_call
                            df_sim.loc[idx_bar, f"TP_Price_Actual{label_suffix_df}"] = tp2_price_open_val_calc_final_call
                            df_sim.loc[idx_bar, f"TP1_Price_Actual{label_suffix_df}"] = tp1_price_open_val_calc_val_call
                            df_sim.loc[idx_bar, f"ATR_At_Entry{label_suffix_df}"] = atr_at_entry_val_open_call
                            df_sim.loc[idx_bar, f"Equity_Before_Open{label_suffix_df}"] = equity_for_lot_open_val_call
                            df_sim.loc[idx_bar, f"Is_Reentry{label_suffix_df}"] = (entry_type_open_val_call == "Re-Entry")
                            df_sim.loc[idx_bar, f"Forced_Entry{label_suffix_df}"] = (entry_type_open_val_call == "Forced")
                            df_sim.loc[idx_bar, f"Meta_Proba_TP{label_suffix_df}"] = meta_proba_open_val_call
                            df_sim.loc[idx_bar, f"Active_Model{label_suffix_df}"] = model_key_open_val_call if model_key_open_val_call is not None else "NONE_SELECTED"
                            df_sim.loc[idx_bar, f"Model_Confidence{label_suffix_df}"] = model_conf_open_val_call
                            df_sim.loc[idx_bar, f"Entry_Gain_Z{label_suffix_df}"] = new_order_instance_val_call.entry_gain_z
                            df_sim.loc[idx_bar, f"Entry_MACD_Smooth{label_suffix_df}"] = new_order_instance_val_call.entry_macd_smooth
                            df_sim.loc[idx_bar, f"Entry_Candle_Ratio{label_suffix_df}"] = new_order_instance_val_call.entry_candle_ratio
                            df_sim.loc[idx_bar, f"Entry_ADX{label_suffix_df}"] = new_order_instance_val_call.entry_adx
                            df_sim.loc[idx_bar, f"Entry_Volatility_Index{label_suffix_df}"] = new_order_instance_val_call.entry_volatility_index
                        else: # pragma: no cover
                            blocked_order_log.append({"timestamp": now_bar, "reason": f"LOT_SIZE_MIN ({final_lot_new_val_call:.2f} < {config_obj.min_lot})", "side": side, "signal_score": row_data_bar.get('Signal_Score', 0.0)})
                elif block_reason_open_val_call != "ALLOWED" and block_reason_open_val_call != "NO_VALID_SIGNAL": # Log other block reasons
                    blocked_order_log.append({"timestamp": now_bar, "reason": block_reason_open_val_call, "side": side, "signal_score": row_data_bar.get('Signal_Score', 0.0)})

            # --- Update Bar-End States ---
            if order_opened_this_bar or any(o.closed for o in active_orders) or active_orders: # Reset if any trade activity
                bars_since_last_trade = 0
            else: # Increment if no trade activity
                bars_since_last_trade += 1

            equity_tracker['current_equity'] = equity_at_bar_start + equity_change_this_bar
            equity_tracker['peak_equity'] = max(equity_tracker['peak_equity'], equity_tracker['current_equity'])
            df_sim.loc[idx_bar, f"Max_Drawdown_At_Point{label_suffix_df}"] = max_drawdown_pct_overall
            df_sim.loc[idx_bar, f"Equity_Realistic{label_suffix_df}"] = equity_tracker['current_equity']
            df_sim.loc[idx_bar, f"Active_Order_Count{label_suffix_df}"] = len(active_orders)
            equity_tracker['history'][now_bar] = equity_tracker['current_equity']

            prev_risk_mode_loop = current_risk_mode
            if consecutive_losses_runtime >= config_obj.recovery_mode_consecutive_losses:
                current_risk_mode = "recovery"
            else:
                current_risk_mode = "normal"
            if prev_risk_mode_loop != current_risk_mode: # pragma: no cover
                sim_logger.info(f"Risk mode changed from '{prev_risk_mode_loop}' to '{current_risk_mode}' for next bar (ConsecLosses: {consecutive_losses_runtime}).")
            df_sim.loc[idx_bar, f"Risk_Mode{label_suffix_df}"] = current_risk_mode
            current_bar_idx += 1
    except Exception as e_loop_main_sim_final_run_full_outer_v2: # pragma: no cover
        sim_logger.critical(f"   (CRITICAL) Outer error in simulation loop for {label}: {e_loop_main_sim_final_run_full_outer_v2}", exc_info=True)
        error_in_loop_runtime = True
        if run_summary and isinstance(run_summary, dict): run_summary["error_msg"] = str(e_loop_main_sim_final_run_full_outer_v2)

    sim_logger.info(f"Simulation loop finished for {label}. Finalizing remaining orders...")
    if active_orders: # pragma: no cover
        sim_logger.info(f"  (End of Period) Closing {len(active_orders)} remaining open orders for {label}...")
        end_time_eop_final_val = df_sim.index[-1] if not df_sim.empty else pd.Timestamp.now(tz='UTC')
        end_close_price_eop_final_val = pd.to_numeric(df_sim["Close"].iloc[-1], errors='coerce') if not df_sim.empty else np.nan
        for order_eop_final_loop_val in active_orders:
            if not order_eop_final_loop_val.closed:
                exit_price_actual_eop_final_val = end_close_price_eop_final_val if pd.notna(end_close_price_eop_final_val) else order_eop_final_loop_val.entry_price
                close_trade(order_eop_final_loop_val, config_obj, exit_price_actual_eop_final_val, end_time_eop_final_val, "EndOfPeriod", order_eop_final_loop_val.lot,
                            trade_log_buffer, equity_tracker, run_summary, label)
    active_orders.clear()

    trade_log_df_final_output_full_val = pd.DataFrame(trade_log_buffer)
    sim_logger.info(f"Created trade log DataFrame for {label} with {len(trade_log_df_final_output_full_val)} entries.")
    equity_col_final_df_sim_val = f"Equity_Realistic{label_suffix_df}"
    if equity_col_final_df_sim_val in df_sim.columns:
        df_sim[equity_col_final_df_sim_val] = df_sim[equity_col_final_df_sim_val].ffill().fillna(initial_capital_segment)
        if not df_sim.empty:
            last_idx_sim_final_df_val = df_sim.index[-1]
            df_sim.loc[last_idx_sim_final_df_val, equity_col_final_df_sim_val] = equity_tracker['current_equity']
            if last_idx_sim_final_df_val not in equity_tracker['history']: equity_tracker['history'][last_idx_sim_final_df_val] = equity_tracker['current_equity']
            if equity_tracker['current_equity'] <= 0: # pragma: no cover
                try:
                    first_zero_idx_sim_df_val = df_sim[df_sim[equity_col_final_df_sim_val] <= 0].index[0]
                    df_sim.loc[first_zero_idx_sim_df_val:, equity_col_final_df_sim_val] = 0.0
                except IndexError: pass # No equity <= 0 found

    if run_summary and isinstance(run_summary, dict):
        run_summary.update({ "error_in_loop": error_in_loop_runtime, "kill_switch_activated": kill_switch_activated_runtime, "final_risk_mode": current_risk_mode })
    else: # Should not happen if initialized correctly # pragma: no cover
        run_summary = { "error_in_loop": error_in_loop_runtime, "kill_switch_activated": kill_switch_activated_runtime, "final_risk_mode": current_risk_mode, "total_ib_lot_accumulator": 0.0 }

    sim_logger.info(f"  (Finished) {label} ({side}) simulation complete. Final Equity: ${equity_tracker['current_equity']:.2f}")
    gc.collect()

    return (
        df_sim, trade_log_df_final_output_full_val, equity_tracker['current_equity'],
        equity_tracker['history'], max_drawdown_pct_overall, run_summary,
        blocked_order_log, meta_model_type_used_runtime, meta_meta_model_type_used_runtime,
        kill_switch_activated_runtime, consecutive_losses_runtime, run_summary.get("total_ib_lot_accumulator", 0.0)
    )

# --- Enterprise Export Functions (NEW) ---
def export_trade_log_to_csv(trades: Union[List[Dict[str, Any]], pd.DataFrame], label: str, output_dir: str, config: 'StrategyConfig'): # type: ignore
    export_logger_csv = logging.getLogger(f"{__name__}.export_trade_log_to_csv")
    if not isinstance(output_dir, str) or not os.path.isdir(output_dir): # pragma: no cover
        export_logger_csv.error(f"[Export] Invalid output directory: {output_dir}. Cannot export trade log for '{label}'.")
        return None
    df_trades_export: Optional[pd.DataFrame] = None
    if isinstance(trades, list):
        if not trades: export_logger_csv.warning(f"[Export] No trades in list for '{label}'"); return None
        try: df_trades_export = pd.DataFrame(trades)
        except Exception as e_df_create_export: export_logger_csv.error(f"[Export] Failed to create DataFrame for '{label}': {e_df_create_export}"); return None
    elif isinstance(trades, pd.DataFrame):
        if trades.empty: export_logger_csv.warning(f"[Export] DataFrame empty for '{label}'"); return None
        df_trades_export = trades
    else: export_logger_csv.error(f"[Export] Invalid 'trades' type: {type(trades)} for '{label}'."); return None # pragma: no cover

    timestamp_str_csv = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix_csv = getattr(config, 'trade_log_filename_prefix', "trade_log")
    export_filename_csv = f"{filename_prefix_csv}_{label}_{timestamp_str_csv}.csv"
    export_path_csv = os.path.join(output_dir, export_filename_csv)
    try:
        # Convert datetime columns to string to avoid timezone issues in CSV if any
        for col_dt_export in df_trades_export.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
            if col_dt_export in df_trades_export: df_trades_export[col_dt_export] = df_trades_export[col_dt_export].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        df_trades_export.to_csv(export_path_csv, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
        export_logger_csv.info(f"[Export] Trade log for '{label}' exported to: {export_path_csv} ({len(df_trades_export)} rows)")
        return export_path_csv
    except Exception as e_export_csv: export_logger_csv.error(f"[Export] Failed to export trade log for '{label}' to {export_path_csv}: {e_export_csv}", exc_info=True); return None # pragma: no cover

def export_run_summary_to_json(run_summary_exp: Dict[str, Any], label: str, output_dir: str, config: 'StrategyConfig'): # type: ignore
    export_logger_json = logging.getLogger(f"{__name__}.export_run_summary_to_json")
    if not isinstance(output_dir, str) or not os.path.isdir(output_dir): export_logger_json.error(f"[Export] Invalid output dir for summary '{label}'."); return None # pragma: no cover
    if not isinstance(run_summary_exp, dict) or not run_summary_exp: export_logger_json.warning(f"[Export] No summary data for '{label}'"); return None # pragma: no cover

    timestamp_str_json = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix_json = getattr(config, 'summary_filename_prefix', "run_summary")
    export_filename_json = f"{filename_prefix_json}_{label}_{timestamp_str_json}.json"
    export_path_json = os.path.join(output_dir, export_filename_json)
    try:
        with open(export_path_json, "w", encoding='utf-8') as f_json:
            json.dump(run_summary_exp, f_json, ensure_ascii=False, indent=2, default=simple_converter) # type: ignore
        export_logger_json.info(f"[Export] Run summary for '{label}' exported to: {export_path_json}")
        return export_path_json
    except Exception as e_export_json: export_logger_json.error(f"[Export] Failed to export run summary for '{label}' to {export_path_json}: {e_export_json}", exc_info=True); return None # pragma: no cover

logger.info("Part 9 (Original Part 8): Backtesting Engine (v4.9.23 - Added TSL/BE Helpers & _check_kill_switch) Loaded and Refactored.")
# === END OF PART 9/15 ===
# === START OF PART 10/15 ===
# ==============================================================================
# === PART 10: Walk-Forward Orchestration & Analysis (v4.9.18 - Corrected WFV Type Hint) ===
# ==============================================================================
# <<< MODIFIED: run_all_folds_with_threshold now calls calculate_m1_entry_signals for each test fold >>>
# <<< using the fold-specific configuration before passing data to backtest simulation. >>>
# <<< MODIFIED: Corrected type hint for model_switcher_func_for_wfv to Optional[Callable]. >>>
# <<< MODIFIED: [Patch v4.9.23] plot_equity_curve - Handle identical min/max xlims to prevent UserWarning. >>>
# <<< MODIFIED: [Patch v4.9.24] plot_equity_curve - Added check for MagicMock plt. >>>
# <<< MODIFIED: [Patch v4.9.24] adjust_gain_z_threshold_by_drift - Adjusted capping logic. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.1] Applied specified patches to plot_equity_curve and adjust_gain_z_threshold_by_drift. >>>

import logging  # Already imported
import os  # Already imported
import pandas as pd  # Already imported
import numpy as np  # Already imported
import math  # Already imported
import json  # Already imported
import matplotlib.pyplot as plt  # type: ignore # Already imported
from matplotlib.ticker import FuncFormatter  # type: ignore # Already imported
from scipy.stats import ttest_ind, wasserstein_distance  # For DriftObserver # Already imported
from sklearn.model_selection import TimeSeriesSplit  # For Walk-Forward # Already imported
import gc  # For memory management # Already imported
from typing import Optional, Callable, Any, Dict, List, Tuple # Ensure this is imported, typically in Part 1
from unittest.mock import MagicMock # [Patch v4.9.24] Added for type checking in plot_equity_curve

# StrategyConfig, RiskManager, TradeManager are defined in Part 3
# run_backtest_simulation_v34 and export functions are defined in Part 9
# calculate_m1_entry_signals is in Part 6
# Other helper functions (load_data, etc.) are in their respective parts.


# --- Drift Observer Class (Refined Logging) ---
class DriftObserver:
    def __init__(self, features_to_observe: List[str]):
        self.features = features_to_observe
        self.results: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {} # Fold_num -> Feature -> {metric: value}
        self.logger = logging.getLogger(f"{__name__}.DriftObserver")
        self.logger.info(f"DriftObserver initialized with {len(self.features)} features to observe.")

    def analyze_fold(self, train_df_pd: pd.DataFrame, test_df_pd: pd.DataFrame, fold_num: int, config: 'StrategyConfig'):  # type: ignore
        self.logger.info(f"    (DriftObserver) Analyzing Drift for Fold {fold_num + 1} (M1 Features)...")
        if not isinstance(train_df_pd, pd.DataFrame) or not isinstance(test_df_pd, pd.DataFrame) or train_df_pd.empty or test_df_pd.empty: # pragma: no cover
            self.logger.warning(f"      Skipping Drift for Fold {fold_num + 1}: Invalid/empty data.")
            self.results[fold_num] = {}
            return

        fold_results_drift_analysis: Dict[str, Dict[str, Optional[float]]] = {}
        self.results[fold_num] = fold_results_drift_analysis
        common_features_drift = list(set(train_df_pd.columns) & set(test_df_pd.columns))
        features_to_analyze = [f for f in self.features if f in common_features_drift]

        if not features_to_analyze: # pragma: no cover
            self.logger.warning(f"      No common observed features to analyze Drift for Fold {fold_num + 1}.")
            return

        wasserstein_threshold = getattr(config, 'drift_wasserstein_threshold', 0.1)
        ttest_alpha = getattr(config, 'drift_ttest_alpha', 0.05)
        min_points = getattr(config, 'drift_min_data_points', 10)
        drift_alert_features = getattr(config, 'drift_alert_features', ['Gain_Z', 'ATR_14', 'Candle_Speed', 'RSI'])
        drift_warning_factor = getattr(config, 'drift_warning_factor', 1.5)
        drift_warning_threshold = wasserstein_threshold * drift_warning_factor
        alert_count = 0

        for feature in features_to_analyze:
            feature_metrics: Dict[str, Optional[float]] = {"wasserstein": np.nan, "ttest_stat": np.nan, "ttest_p": np.nan}
            try:
                if not pd.api.types.is_numeric_dtype(train_df_pd[feature]) or \
                   not pd.api.types.is_numeric_dtype(test_df_pd[feature]): # pragma: no cover
                    self.logger.debug(f"         Skipping non-numeric feature for drift: '{feature}'")
                    fold_results_drift_analysis[feature] = feature_metrics
                    continue

                train_series = pd.to_numeric(train_df_pd[feature], errors='coerce').dropna()
                test_series = pd.to_numeric(test_df_pd[feature], errors='coerce').dropna()

                if len(train_series) < min_points or len(test_series) < min_points: # pragma: no cover
                    self.logger.debug(f"         Skipping '{feature}': Insufficient data (Train: {len(train_series)}, Test: {len(test_series)}).")
                    fold_results_drift_analysis[feature] = feature_metrics
                    continue

                w_dist = wasserstein_distance(train_series, test_series)
                feature_metrics["wasserstein"] = w_dist

                log_level_drift = logging.DEBUG  # Default log level for drift info
                drift_msg_prefix = "(Drift Info)"
                if feature in drift_alert_features and w_dist > wasserstein_threshold: # pragma: no cover
                    log_level_drift = logging.WARNING
                    drift_msg_prefix = "[DRIFT ALERT]"
                    alert_count += 1
                elif w_dist > drift_warning_threshold:  # Warning for any feature exceeding this higher threshold # pragma: no cover
                    log_level_drift = logging.WARNING
                    drift_msg_prefix = "(Drift Warning)"
                elif w_dist > wasserstein_threshold:  # Info for other features exceeding base threshold # pragma: no cover
                    log_level_drift = logging.INFO

                if w_dist > wasserstein_threshold:  # Log if above base threshold # pragma: no cover
                    self.logger.log(log_level_drift, f"          {drift_msg_prefix} Feature='{feature}', Wasserstein={w_dist:.4f} (Threshold: {wasserstein_threshold:.2f}) (Fold {fold_num+1})")

                if train_series.var() > 1e-9 and test_series.var() > 1e-9: # Check variance before t-test
                    t_stat, t_p = ttest_ind(train_series, test_series, equal_var=False, nan_policy='omit')
                    feature_metrics["ttest_stat"] = t_stat
                    feature_metrics["ttest_p"] = t_p
                    if t_p < ttest_alpha: # pragma: no cover
                        self.logger.info(f"          (Drift T-test) Feature='{feature}', p-value = {t_p:.4f} (<{ttest_alpha:.2f}) (Fold {fold_num+1})")
            except Exception as e: # pragma: no cover
                self.logger.error(f"      Error calculating drift for '{feature}' in Fold {fold_num + 1}: {e}")
            finally:
                fold_results_drift_analysis[feature] = feature_metrics
        self.logger.info(f"    (DriftObserver) Analysis complete for Fold {fold_num + 1}. Alerts (Wasserstein > Thresh on Alert Features): {alert_count}")

    def get_fold_drift_summary(self, fold_num: int) -> float:
        if fold_num not in self.results:
            return np.nan
        fold_data = self.results[fold_num]
        if not fold_data:
            return np.nan
        w_dists = [res["wasserstein"] for res in fold_data.values() if isinstance(res, dict) and pd.notna(res.get("wasserstein"))]
        return np.mean(w_dists) if w_dists else np.nan # type: ignore

    def summarize_and_save(self, output_dir: str, config: 'StrategyConfig'):  # type: ignore
        self.logger.info("\n(DriftObserver) Summarizing M1 Feature Drift analysis results...")
        if not self.results: # pragma: no cover
            self.logger.warning("No drift results to summarize.")
            return
        wasserstein_threshold = getattr(config, 'drift_wasserstein_threshold', 0.1)
        ttest_alpha = getattr(config, 'drift_ttest_alpha', 0.05)
        summary_data: List[Dict[str, Any]] = []
        for fold_num, fold_data_item in sorted(self.results.items()):
            if not fold_data_item: # pragma: no cover
                continue
            numeric_data = {feat: res for feat, res in fold_data_item.items() if isinstance(res, dict) and pd.notna(res.get("wasserstein"))}
            if not numeric_data: # pragma: no cover
                continue
            w_dists = [res["wasserstein"] for res in numeric_data.values() if res["wasserstein"] is not None] # Ensure not None
            p_vals = [res["ttest_p"] for res in numeric_data.values() if res.get("ttest_p") is not None and pd.notna(res.get("ttest_p"))]
            summary_data.append({
                "Fold": fold_num + 1, "Mean_Wasserstein": np.mean(w_dists) if w_dists else np.nan,
                "Max_Wasserstein": np.max(w_dists) if w_dists else np.nan,
                "Drift_Features_Wasserstein": sum(1 for d in w_dists if d > wasserstein_threshold),
                "Drift_Features_Ttest": sum(1 for p in p_vals if p < ttest_alpha),
                "Total_Analyzed_Numeric_Features": len(w_dists)
            })
        if not summary_data: # pragma: no cover
            self.logger.warning("No fold data for drift summary CSV.")
            return
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, f"drift_summary_m1_{config.output_dir_name}.csv")
        try:
            summary_df.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.4f")
            self.logger.info(f"  Saved M1 drift summary (CSV): {csv_path}")
            self.logger.info("--- Drift Summary per Fold ---\n" + summary_df.to_string(index=False, float_format="%.4f"))
        except Exception as e: # pragma: no cover
            self.logger.error(f"  Failed to save drift summary CSV: {e}", exc_info=True)

    def export_fold_summary(self, output_dir: str, fold_num: int):
        if fold_num not in self.results or not self.results[fold_num]: # pragma: no cover
            return
        fold_data = self.results[fold_num]
        fold_summary_list: List[Dict[str, Any]] = [{'feature': f, **m} for f, m in fold_data.items() if isinstance(m, dict)]
        if not fold_summary_list: # pragma: no cover
            return
        fold_df = pd.DataFrame(fold_summary_list)
        cols_order = ['feature', 'wasserstein', 'ttest_stat', 'ttest_p']
        fold_df = fold_df[[c for c in cols_order if c in fold_df.columns]] # Ensure columns exist before selection
        path = os.path.join(output_dir, f"drift_details_fold{fold_num+1}.csv")
        try:
            fold_df.to_csv(path, index=False, float_format="%.4f")
            self.logger.debug(f"Exported Drift for Fold {fold_num+1}: {path}")
        except Exception as e: # pragma: no cover
            self.logger.error(f"Failed to export Drift for Fold {fold_num+1}: {e}")

    def save(self, filepath: str): # pragma: no cover
        self.logger.info(f"   (DriftObserver) Skipping save of DriftObserver object to {filepath}.")

    def load(self, filepath: str) -> bool: # Added return type # pragma: no cover
        self.logger.info(f"   (DriftObserver) Skipping load of DriftObserver object from {filepath}.")
        self.results = {} # Reset results if load is skipped
        return False

# --- Performance Metrics Calculation (Refined Hit Rate Definitions) ---
def calculate_metrics(
    config: 'StrategyConfig', trade_log_df: Optional[pd.DataFrame], final_equity: float,  # type: ignore
    equity_history_segment: Optional[Union[Dict[pd.Timestamp, float], pd.Series]], label: str = "",
    model_type_l1: str = "N/A", model_type_l2: str = "N/A",
    run_summary: Optional[Dict[str, Any]] = None, ib_lot_accumulator: float = 0.0
) -> Dict[str, Any]:
    metrics_logger = logging.getLogger(f"{__name__}.calculate_metrics.{label}")
    metrics: Dict[str, Any] = {}
    metrics_logger.info(f"  (Metrics) Calculating full metrics for: '{label}'...")

    initial_capital = config.initial_capital
    metrics[f"{label} Initial Capital (USD)"] = initial_capital

    default_trade_metrics: Dict[str, Any] = {
        "Total Trades (Full)": 0, "Total Net Profit (USD)": 0.0, "Gross Profit (USD)": 0.0, "Gross Loss (USD)": 0.0,
        "Profit Factor": 0.0, "Average Trade (Full) (USD)": 0.0, "Max Trade Win (Full) (USD)": 0.0,
        "Max Trade Loss (Full) (USD)": 0.0, "Total Wins (Full)": 0, "Total Losses (Full)": 0,
        "Win Rate (Full) (%)": 0.0, "Average Win (Full) (USD)": 0.0, "Average Loss (Full) (USD)": 0.0,
        "Payoff Ratio (Full)": 0.0, "BE-SL Exits (Full)": 0, "Expectancy (Full) (USD)": 0.0,
        "TP Rate (Full Trades) (%)": 0.0, # Rate of full trades that hit TP2
        "Re-Entry Trades (Full)": 0, "Forced Entry Trades (Full)": 0,
        "Partial TP Events": 0, # Total number of PTP events (can be > num_full_trades)
        "Trades with PTP": 0, # Number of unique orders that had at least one PTP
        "Entry Count (Total Actions)": 0, # Total rows in log_df (includes PTP actions)
        "Unique Orders Initiated": 0, # Based on unique entry_idx
        "TP1 Hit Rate (vs Unique Orders) (%)": 0.0, # Unique orders hitting at least PTP1 or TP2
        "TP2 Hit Rate (vs Unique Orders) (%)": 0.0, # Unique orders hitting TP2 (full TP)
        "SL Hit Rate (Full Trades vs Unique Orders) (%)": 0.0, # Unique orders hitting SL (full SL)
        "Total Lots Traded (IB Accumulator)": 0.0, "IB Commission Estimate (USD)": 0.0,
    }
    for key, val in default_trade_metrics.items():
        metrics[f"{label} {key}"] = val

    if trade_log_df is None or not isinstance(trade_log_df, pd.DataFrame) or trade_log_df.empty:
        metrics_logger.warning(f"    No trades logged for '{label}'.")
    else:
        log_df = trade_log_df.copy()
        # Ensure necessary columns are correct type or have defaults
        log_df["pnl_usd_net"] = pd.to_numeric(log_df["pnl_usd_net"], errors='coerce').fillna(0.0)
        log_df["exit_reason"] = log_df["exit_reason"].astype(str).fillna("N/A")
        log_df["is_partial_tp_event"] = log_df.get("is_partial_tp_event", pd.Series(False, index=log_df.index)).astype(bool)
        log_df["entry_idx"] = log_df.get("entry_idx", pd.Series(range(len(log_df)))) # Fallback if entry_idx is missing

        # Full trades are those that are not partial TP events (i.e., final exits of an order)
        full_trades = log_df[~log_df["is_partial_tp_event"]].copy()
        num_full_trades = len(full_trades)
        metrics[f"{label} Total Trades (Full)"] = num_full_trades
        metrics[f"{label} Total Net Profit (USD)"] = log_df["pnl_usd_net"].sum() # Sum PnL from all events
        metrics[f"{label} Partial TP Events"] = log_df["is_partial_tp_event"].sum()
        if 'entry_idx' in log_df.columns and 'is_partial_tp_event' in log_df.columns:
            metrics[f"{label} Trades with PTP"] = log_df[log_df['is_partial_tp_event']]['entry_idx'].nunique()
        metrics[f"{label} Entry Count (Total Actions)"] = len(log_df) # Total rows in log
        unique_orders_initiated = log_df['entry_idx'].nunique() if 'entry_idx' in log_df else num_full_trades # Fallback if no entry_idx
        metrics[f"{label} Unique Orders Initiated"] = unique_orders_initiated

        if num_full_trades > 0:
            pnl = full_trades["pnl_usd_net"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            metrics[f"{label} Gross Profit (USD)"] = wins.sum()
            metrics[f"{label} Gross Loss (USD)"] = losses.sum() # This will be negative or zero
            gp = metrics[f"{label} Gross Profit (USD)"]
            gl_abs = abs(metrics[f"{label} Gross Loss (USD)"])
            metrics[f"{label} Profit Factor"] = gp / gl_abs if gl_abs > 1e-9 else (np.inf if gp > 0 else 0.0)
            metrics[f"{label} Average Trade (Full) (USD)"] = pnl.mean()
            metrics[f"{label} Max Trade Win (Full) (USD)"] = wins.max() if not wins.empty else 0.0
            metrics[f"{label} Max Trade Loss (Full) (USD)"] = losses.min() if not losses.empty else 0.0 # This will be negative
            win_count = len(wins)
            loss_count = len(losses)
            metrics[f"{label} Total Wins (Full)"] = win_count
            metrics[f"{label} Total Losses (Full)"] = loss_count
            metrics[f"{label} Win Rate (Full) (%)"] = (win_count / num_full_trades) * 100.0 if num_full_trades > 0 else 0.0
            avg_win = wins.mean() if win_count > 0 else 0.0
            avg_loss = losses.mean() if loss_count > 0 else 0.0 # This will be negative
            metrics[f"{label} Average Win (Full) (USD)"] = avg_win
            metrics[f"{label} Average Loss (Full) (USD)"] = avg_loss
            avg_loss_abs = abs(avg_loss)
            metrics[f"{label} Payoff Ratio (Full)"] = avg_win / avg_loss_abs if avg_loss_abs > 1e-9 else (np.inf if avg_win > 0 else 0.0)
            metrics[f"{label} BE-SL Exits (Full)"] = (full_trades["exit_reason"].str.upper() == "BE-SL").sum()
            tp_hits_full_trades = (full_trades["exit_reason"].str.upper() == "TP").sum() # Full TP2 hits
            metrics[f"{label} TP Rate (Full Trades) (%)"] = (tp_hits_full_trades / num_full_trades) * 100.0 if num_full_trades > 0 else 0.0
            wr_dec = metrics[f"{label} Win Rate (Full) (%)"] / 100.0
            metrics[f"{label} Expectancy (Full) (USD)"] = (avg_win * wr_dec) + (avg_loss * (1.0 - wr_dec)) # avg_loss is negative
            if "Is_Reentry" in full_trades.columns: # pragma: no cover
                metrics[f"{label} Re-Entry Trades (Full)"] = full_trades["Is_Reentry"].astype(bool).sum()
            if "Is_Forced_Entry" in full_trades.columns: # pragma: no cover
                metrics[f"{label} Forced Entry Trades (Full)"] = full_trades["Is_Forced_Entry"].astype(bool).sum()

            # TP1 Hit Rate: An order hit TP1 if it had a PTP event OR it went straight to TP2 (full TP)
            orders_with_ptp_event = set(log_df[log_df['is_partial_tp_event']]['entry_idx'])
            orders_with_full_tp = set(full_trades[full_trades['exit_reason'].str.upper() == 'TP']['entry_idx'])
            tp1_hit_orders_count = len(orders_with_ptp_event.union(orders_with_full_tp))
            metrics[f"{label} TP1 Hit Rate (vs Unique Orders) (%)"] = (tp1_hit_orders_count / unique_orders_initiated) * 100.0 if unique_orders_initiated > 0 else 0.0

            # TP2 Hit Rate: An order hit TP2 if its final exit reason (from full_trades) was 'TP'
            tp2_hit_orders_count = len(orders_with_full_tp)
            metrics[f"{label} TP2 Hit Rate (vs Unique Orders) (%)"] = (tp2_hit_orders_count / unique_orders_initiated) * 100.0 if unique_orders_initiated > 0 else 0.0

            # SL Hit Rate: An order hit SL if its final exit reason (from full_trades) was 'SL'
            sl_hit_orders_count = len(full_trades[full_trades['exit_reason'].str.upper() == 'SL']['entry_idx'].unique())
            metrics[f"{label} SL Hit Rate (Full Trades vs Unique Orders) (%)"] = (sl_hit_orders_count / unique_orders_initiated) * 100.0 if unique_orders_initiated > 0 else 0.0

    metrics[f"{label} Total Lots Traded (IB Accumulator)"] = ib_lot_accumulator
    metrics[f"{label} IB Commission Estimate (USD)"] = ib_lot_accumulator * getattr(config, 'ib_commission_per_lot', 7.0)
    metrics[f"{label} Final Equity (USD)"] = final_equity
    if initial_capital > 1e-9:
        metrics[f"{label} Return (%)"] = ((final_equity - initial_capital) / initial_capital) * 100.0
        metrics[f"{label} Absolute Profit (USD)"] = final_equity - initial_capital
    else: # pragma: no cover
        metrics[f"{label} Return (%)"] = 0.0
        metrics[f"{label} Absolute Profit (USD)"] = 0.0

    # Equity based metrics (Sharpe, Sortino, Max Drawdown)
    equity_series: Optional[pd.Series] = None
    if isinstance(equity_history_segment, pd.Series):
        equity_series = equity_history_segment.copy()
    elif isinstance(equity_history_segment, dict) and equity_history_segment:
        try:
            equity_series = pd.Series({pd.to_datetime(k, errors='coerce'): v for k, v in equity_history_segment.items()}).dropna().sort_index()
            if not equity_series.empty:
                equity_series = equity_series[~equity_series.index.duplicated(keep='last')] # Ensure unique index
            else: # pragma: no cover
                equity_series = None
        except Exception as e: # pragma: no cover
            metrics_logger.error(f"Error converting equity history: {e}")
            equity_series = None

    if equity_series is not None and len(equity_series) > 1:
        try:
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max.replace(0, np.nan) # Avoid division by zero if peak is 0
            max_dd_val = drawdown.min()
            metrics[f"{label} Max Drawdown (Equity based) (%)"] = abs(max_dd_val * 100.0) if pd.notna(max_dd_val) else 0.0

            # Risk-adjusted return metrics (Sharpe, Sortino, Calmar) - require daily-like returns
            if isinstance(equity_series.index, pd.DatetimeIndex):
                equity_resampled = equity_series.resample('B').last().ffill().dropna() # Resample to business days
                if len(equity_resampled) > 20: # Need enough data points for meaningful stats
                    daily_ret = equity_resampled.pct_change().dropna()
                    if not daily_ret.empty and initial_capital > 1e-9 and daily_ret.std() > 1e-9:
                        # Calculate annualized return
                        total_return_overall = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1 if not equity_series.empty and equity_series.iloc[0] != 0 else 0
                        num_days_total = (equity_series.index[-1] - equity_series.index[0]).days if not equity_series.empty else 0
                        num_years = num_days_total / 365.25 if num_days_total > 0 else 1.0 / 252.0 # Avoid div by zero, assume 1 day if less
                        annualized_return = ((1 + total_return_overall) ** (1 / num_years)) - 1 if (1 + total_return_overall) > 0 else -1.0

                        ann_std = daily_ret.std(ddof=1) * math.sqrt(252) # Annualized std dev
                        metrics[f"{label} Sharpe Ratio (approx)"] = annualized_return / ann_std if ann_std > 1e-9 else (np.inf if annualized_return > 0 else 0.0)

                        downside_ret = daily_ret[daily_ret < 0]
                        if not downside_ret.empty and downside_ret.std() > 1e-9:
                            downside_std = downside_ret.std(ddof=1) * math.sqrt(252)
                            metrics[f"{label} Sortino Ratio (approx)"] = annualized_return / downside_std if downside_std > 1e-9 else (np.inf if annualized_return > 0 else 0.0)
                        elif annualized_return >= 0: # No downside returns, and positive overall return
                            metrics[f"{label} Sortino Ratio (approx)"] = np.inf
                        else: # No downside returns, but negative overall return # pragma: no cover
                            metrics[f"{label} Sortino Ratio (approx)"] = -np.inf


                        max_dd_pct_for_calmar = metrics[f"{label} Max Drawdown (Equity based) (%)"]
                        metrics[f"{label} Calmar Ratio (approx)"] = (annualized_return * 100.0) / max_dd_pct_for_calmar if max_dd_pct_for_calmar > 1e-9 else (np.inf if annualized_return > 0 else 0.0)
        except Exception as e: # pragma: no cover
            metrics_logger.error(f"Error calculating equity/drawdown/ratio metrics: {e}", exc_info=True)
    metrics_logger.info(f"  (Metrics) Finished calculating full metrics for: '{label}'.")
    return metrics

# --- Equity Curve Plotting (Fuller Implementation) ---
def plot_equity_curve(
    config: 'StrategyConfig', equity_series_data: Optional[Union[Dict[pd.Timestamp, float], pd.Series]], title: str,  # type: ignore
    output_dir_plot: str, filename_suffix_plot: str, fold_boundaries: Optional[List[pd.Timestamp]] = None
):
    plot_logger = logging.getLogger(f"{__name__}.plot_equity_curve.{filename_suffix_plot}")
    plot_logger.info(f"\n--- (Plotting) Plotting Equity Curve: {title} ---")
    initial_capital_plot = config.initial_capital
    equity_series_for_plot: Optional[pd.Series] = None

    # Convert input data to a sorted pandas Series with DatetimeIndex
    if isinstance(equity_series_data, dict):
        if equity_series_data: # pragma: no cover
            try:
                equity_series_for_plot = pd.Series({pd.to_datetime(k, errors='coerce'): v for k, v in equity_series_data.items()}).dropna().sort_index()
                if not equity_series_for_plot.empty:
                    equity_series_for_plot = equity_series_for_plot[~equity_series_for_plot.index.duplicated(keep='last')] # Deduplicate index
                else:
                    equity_series_for_plot = None # Ensure it's None if all NaT or empty
            except Exception as e: # pragma: no cover
                plot_logger.error(f"Error converting equity dict for plot: {e}")
                equity_series_for_plot = None
    elif isinstance(equity_series_data, pd.Series):
        equity_series_for_plot = equity_series_data.copy()
        if not isinstance(equity_series_for_plot.index, pd.DatetimeIndex): # pragma: no cover
            try:
                equity_series_for_plot.index = pd.to_datetime(equity_series_for_plot.index, errors='coerce')
                equity_series_for_plot = equity_series_for_plot[equity_series_for_plot.index.notna()] # Remove NaT indices
                if equity_series_for_plot.empty:
                    equity_series_for_plot = None
            except Exception as e: # pragma: no cover
                plot_logger.error(f"Error converting equity index for plot: {e}")
                equity_series_for_plot = None
        if equity_series_for_plot is not None and isinstance(equity_series_for_plot.index, pd.DatetimeIndex):
            if not equity_series_for_plot.index.is_monotonic_increasing: # pragma: no cover
                equity_series_for_plot.sort_index(inplace=True)
            if equity_series_for_plot.index.has_duplicates: # pragma: no cover
                equity_series_for_plot = equity_series_for_plot[~equity_series_for_plot.index.duplicated(keep='last')]

    if plt is None: # pragma: no cover
        plot_logger.warning("Matplotlib (plt) is None. Skipping actual plot saving.")
        return

    # <<< MODIFIED: [Patch v4.9.24] Added check for MagicMock plt. >>>
    # <<< MODIFIED: [Patch AI Studio v4.9.1] Applied specified patches to plot_equity_curve. >>>
    if isinstance(plt, MagicMock): # pragma: no cover
        # Check if plt.subplots.return_value is also a MagicMock or not a (fig, ax) tuple
        # This is common in testing environments where plt is fully mocked.
        if isinstance(plt.subplots.return_value, MagicMock) or \
           not (isinstance(plt.subplots.return_value, tuple) and len(plt.subplots.return_value) == 2):
            plot_logger.warning("   [Patch AI Studio v4.9.1] plt is MagicMock and plt.subplots.return_value is not a (fig, ax) tuple. Skipping plot generation in test context.")
            return
    # <<< END OF MODIFIED [Patch AI Studio v4.9.1] >>>

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_filename = os.path.join(output_dir_plot, f"equity_curve_{filename_suffix_plot}.png")

    if equity_series_for_plot is None or equity_series_for_plot.empty: # pragma: no cover
        plot_logger.warning(f"   No valid equity data for '{title}'. Plotting baseline initial capital line only.")
        ax.axhline(initial_capital_plot, color='red', linestyle=":", linewidth=1.5, label=f"Initial Capital (${initial_capital_plot:,.2f})")
        ax.set_title(f"{title} (No Trade Data)", fontsize=14)
        ax.set_ylabel("Equity (USD)", fontsize=12)
        ax.set_xlabel("Date (No Trades)", fontsize=12)
    else:
        try:
            equity_series_for_plot.plot(ax=ax, label="Equity", legend=True, grid=True, linewidth=1.5, color="blue", alpha=0.8)
            ax.axhline(initial_capital_plot, color='red', linestyle=":", linewidth=1.5, label=f"Initial Capital (${initial_capital_plot:,.2f})")

            if fold_boundaries and isinstance(fold_boundaries, list):
                valid_bounds = pd.to_datetime(fold_boundaries, errors='coerce').dropna().tolist()
                if len(valid_bounds) >= 1: # pragma: no cover
                    plotted_labels: set[str] = set() # To avoid duplicate legend entries
                    # Plot start and end of the entire test period
                    if equity_series_for_plot.index.is_monotonic_increasing and not equity_series_for_plot.empty:
                        ax.axvline(equity_series_for_plot.index[0], color="darkgreen", linestyle="--", linewidth=1.2, label="Test Start")
                        plotted_labels.add("Test Start")

                    for i, bound_ts in enumerate(valid_bounds):
                        label_bound = f"End Fold {i+1}"
                        if i == 0 and len(valid_bounds) > 1 : # If it's the first boundary of multiple, consider it start of test fold
                            label_bound = "Start Test Fold 1" # Or similar, depends on WFV structure
                        plot_label_str = label_bound if label_bound not in plotted_labels else "_nolegend_"
                        ax.axvline(bound_ts, color="grey", linestyle="--", linewidth=1, label=plot_label_str)
                        plotted_labels.add(label_bound)

                    if equity_series_for_plot.index.is_monotonic_increasing and not equity_series_for_plot.empty:
                        ax.axvline(equity_series_for_plot.index[-1], color="purple", linestyle="--", linewidth=1.2, label="Test End")
                        plotted_labels.add("Test End")

                    # <<< MODIFIED: [Patch v4.9.23] Handle identical min/max xlims to prevent UserWarning. >>>
                    # <<< MODIFIED: [Patch AI Studio v4.9.1] Applied specified patches to plot_equity_curve. >>>
                    all_times_for_xlim = pd.to_datetime(list(equity_series_for_plot.index) + valid_bounds, errors='coerce').dropna()
                    if not all_times_for_xlim.empty:
                        min_xlim = all_times_for_xlim.min()
                        max_xlim = all_times_for_xlim.max()
                        if min_xlim == max_xlim: # Handle case where all times are identical
                            plot_logger.debug(f"   [Patch AI Studio v4.9.1] xlim min and max are identical ({min_xlim}). Adjusting max_xlim for plotting.")
                            # Try to infer a reasonable delta from data, fallback to 1 day
                            time_diffs_plot = equity_series_for_plot.index.to_series().diff().median() if len(equity_series_for_plot.index) > 1 else pd.Timedelta(minutes=1)
                            if pd.isna(time_diffs_plot) or time_diffs_plot.total_seconds() == 0:
                                time_diffs_plot = pd.Timedelta(days=1) # Fallback delta
                            max_xlim = min_xlim + time_diffs_plot
                            plot_logger.debug(f"      New max_xlim: {max_xlim} (Delta used: {time_diffs_plot})")
                        ax.set_xlim(min_xlim, max_xlim)
                    # <<< END OF MODIFIED [Patch AI Studio v4.9.1] >>>

            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_ylabel("Equity (USD)", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
        except Exception as e: # pragma: no cover
            plot_logger.error(f"   Error during main plot elements for '{title}': {e}", exc_info=True)

    if ax.get_legend_handles_labels()[1]: # Only add legend if there are labels
        ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='x', rotation=15, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    try:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}")) # type: ignore
    except Exception: # pragma: no cover
        pass # If FuncFormatter fails for any reason, just use default
    plt.tight_layout(pad=1.5)
    try:
        plt.savefig(plot_filename, dpi=200, bbox_inches="tight")
        plot_logger.info(f"   Saved equity curve plot: {plot_filename}")
    except Exception as e: # pragma: no cover
        plot_logger.error(f"   Failed to save equity plot '{plot_filename}': {e}", exc_info=True)
    finally:
        plt.close(fig)


# --- Log Analysis Functions (Placeholders) ---
def load_trade_log(log_file_path: str) -> Optional[pd.DataFrame]: # pragma: no cover
    log_analysis_logger = logging.getLogger(f"{__name__}.load_trade_log")
    log_analysis_logger.info(f"   (Log Analysis) Attempting to load trade log: {log_file_path}")
    if os.path.exists(log_file_path):
        try:
            df = pd.read_csv(log_file_path, compression='gzip' if log_file_path.endswith('.gz') else None)
            log_analysis_logger.info(f"      Successfully loaded trade log with {len(df)} entries.")
            return df
        except Exception as e:
            log_analysis_logger.error(f"      Error loading trade log: {e}")
            return None
    else:
        log_analysis_logger.warning("      Trade log file not found.")
        return None

def run_log_analysis_pipeline(log_file_path: str, output_dir_log_analysis: str, config: 'StrategyConfig', suffix_log_analysis: str) -> Optional[Dict[str, Any]]: # Added config
    log_analysis_logger = logging.getLogger(f"{__name__}.run_log_analysis_pipeline")
    consecutive_loss_config_val = config.recovery_mode_consecutive_losses # Get from config
    log_analysis_logger.info(f"   (Log Analysis) Running analysis pipeline for: {log_file_path} (Suffix: {suffix_log_analysis}, ConsecLossThresh: {consecutive_loss_config_val})")
    log_analysis_logger.warning("   (Log Analysis) 'run_log_analysis_pipeline' is a placeholder. No detailed analysis performed.")
    return {"status": "placeholder_executed", "log_file": os.path.basename(log_file_path), "consecutive_loss_threshold_used": consecutive_loss_config_val}


# --- Dynamic Parameter Adjustment Helper (Now takes StrategyConfig) ---
# <<< MODIFIED: [Patch v4.9.24] adjust_gain_z_threshold_by_drift - Adjusted capping logic. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.1] Applied specified patches to adjust_gain_z_threshold_by_drift. >>>
def adjust_gain_z_threshold_by_drift(
    base_threshold: float,
    drift_score: Optional[float],
    config: 'StrategyConfig',  # type: ignore
    adjustment_factor: float = 0.1, # Default factor, could be from config too
    max_adjustment_pct: float = 0.5 # Default max % change, could be from config
) -> float:
    adj_logger = logging.getLogger(f"{__name__}.adjust_gain_z_threshold_by_drift")
    if drift_score is None or pd.isna(drift_score):
        adj_logger.debug("Drift score is None/NaN, returning base threshold.")
        return base_threshold

    drift_sensitivity = config.drift_adjustment_sensitivity
    max_threshold_val = config.drift_max_gain_z_thresh
    min_threshold_val = config.drift_min_gain_z_thresh

    # Calculate raw adjustment based on drift score and sensitivity
    adjustment = drift_score * adjustment_factor * drift_sensitivity
    adjusted_threshold_raw = base_threshold * (1 + adjustment)
    adj_logger.debug(f"   [Patch AI Studio v4.9.1] Base: {base_threshold:.4f}, Drift: {drift_score:.3f}, Raw Adjusted: {adjusted_threshold_raw:.4f} (Adjustment: {adjustment:.4f})")

    # Apply percentage-based cap on the change from base_threshold
    max_abs_change_from_base = abs(base_threshold * max_adjustment_pct)
    change_from_base = adjusted_threshold_raw - base_threshold

    if abs(change_from_base) > max_abs_change_from_base:
        adjusted_threshold_capped_pct = base_threshold + (np.sign(change_from_base) * max_abs_change_from_base)
        adj_logger.debug(f"   [Patch AI Studio v4.9.1] Threshold adjustment capped by max_adjustment_pct ({max_adjustment_pct*100}% = {max_abs_change_from_base:.4f}). Capped value: {adjusted_threshold_capped_pct:.4f} (was {adjusted_threshold_raw:.4f})")
    else:
        adjusted_threshold_capped_pct = adjusted_threshold_raw
        adj_logger.debug(f"   [Patch AI Studio v4.9.1] Threshold adjustment within max_adjustment_pct. Value remains: {adjusted_threshold_capped_pct:.4f}")

    # Apply absolute min/max caps
    final_threshold = np.clip(adjusted_threshold_capped_pct, min_threshold_val, max_threshold_val)
    adj_logger.debug(f"   [Patch AI Studio v4.9.1] Final threshold after min/max clipping ({min_threshold_val:.4f} to {max_threshold_val:.4f}): {final_threshold:.4f} (from capped_pct: {adjusted_threshold_capped_pct:.4f})")

    if not math.isclose(final_threshold, base_threshold): # pragma: no cover
        adj_logger.info(f"Gain_Z threshold adjusted due to drift. Base: {base_threshold:.3f}, Drift: {drift_score:.3f}, Adjusted: {final_threshold:.3f}")
    return final_threshold
# <<< END OF MODIFIED [Patch AI Studio v4.9.1] >>>


# --- Walk-Forward Orchestration (Refactored to use injected objects and config) ---
def run_all_folds_with_threshold(
    config_obj: 'StrategyConfig', risk_manager_obj: 'RiskManager', trade_manager_obj: 'TradeManager',  # type: ignore
    df_m1_final_for_wfv: pd.DataFrame, output_dir_for_wfv: str,
    available_models_for_wfv: Optional[Dict[str, Any]] = None,
    model_switcher_func_for_wfv: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[Optional[str], Optional[float]]]] = None, # <<< MODIFIED: Type hint
    drift_observer_for_wfv: Optional[DriftObserver] = None, current_l1_threshold_override_for_wfv: Optional[float] = None,
    fund_profile_for_wfv: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], pd.DataFrame, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Optional[pd.DataFrame], str, str, float]:
    wfv_logger = logging.getLogger(f"{__name__}.run_all_folds_with_threshold")

    n_splits_wfv = config_obj.n_walk_forward_splits
    initial_capital_wfv = config_obj.initial_capital
    default_l1_thresh_wfv = config_obj.meta_min_proba_thresh
    current_fund_profile_wfv = fund_profile_for_wfv if fund_profile_for_wfv else \
        getattr(config_obj, 'fund_profiles', {}).get(
            config_obj.default_fund_name,
            {"name": "FALLBACK_FUND", "risk": config_obj.risk_per_trade, "mm_mode": "balanced"}
        )
    fund_name_for_wfv_log = current_fund_profile_wfv.get("name", "UnnamedFund_WFV")
    config_obj.current_fund_name_for_logging = fund_name_for_wfv_log # Set for deeper logging context

    is_data_prep_wfv = (available_models_for_wfv is None and model_switcher_func_for_wfv is None)
    run_label_wfv = f"(Prep Data - Fund: {fund_name_for_wfv_log})" if is_data_prep_wfv else f"(Final Run - Fund: {fund_name_for_wfv_log})"

    if not output_dir_for_wfv or not os.path.isdir(output_dir_for_wfv): # pragma: no cover
        wfv_logger.critical(f"      [Runner {run_label_wfv}] Output directory '{output_dir_for_wfv}' invalid.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0
    l1_thresh_to_use_wfv = current_l1_threshold_override_for_wfv if current_l1_threshold_override_for_wfv is not None else default_l1_thresh_wfv
    wfv_logger.info(f"      [Runner {run_label_wfv}] Starting Full WF Sim (L1_Th={l1_thresh_to_use_wfv:.2f}) for {n_splits_wfv} folds.")

    if df_m1_final_for_wfv is None or df_m1_final_for_wfv.empty: # pragma: no cover
        wfv_logger.error(f"      [Runner {run_label_wfv}] M1 Data (df_m1_final_for_wfv) is empty or None. Cannot proceed.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0
    if not is_data_prep_wfv and (model_switcher_func_for_wfv is None or available_models_for_wfv is None or not available_models_for_wfv.get('main')): # pragma: no cover
        wfv_logger.error(f"      [Runner {run_label_wfv}] Model switcher or available models (main) not provided for non-data-prep run. Cannot proceed.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    tscv_wfv = TimeSeriesSplit(n_splits=n_splits_wfv)
    all_fold_results_list: List[pd.DataFrame] = []
    all_trade_logs_list: List[pd.DataFrame] = []
    all_equity_histories_dict: Dict[str, Dict[pd.Timestamp, float]] = {}
    all_fold_metrics_list: List[Dict[str, Any]] = []
    all_blocked_logs_list: List[Dict[str, Any]] = []
    previous_fold_metrics_data: Optional[Dict[str, Any]] = None # For potential future use (e.g., adapting params based on prev fold)
    model_type_l1_overall = "Switcher" if model_switcher_func_for_wfv and available_models_for_wfv else "SignalOnly"
    model_type_l2_overall = "N/A" # Placeholder, not used in this version
    first_fold_test_data_output: Optional[pd.DataFrame] = None # For SHAP analysis outside WFV on first fold's test data
    total_ib_lot_accumulator_overall: float = 0.0

    # Reset RiskManager and TradeManager states for this fund's WFV run
    risk_manager_obj.dd_peak = None # Will be re-initialized by first chained capital
    risk_manager_obj.soft_kill_active = False
    trade_manager_obj.last_trade_time = None
    trade_manager_obj.consecutive_forced_losses = 0
    wfv_logger.info(f"   RiskManager and TradeManager states reset for WFV execution of Fund: {fund_name_for_wfv_log}")

    # Initialize chained capital and states for BUY and SELL simulations
    chained_capital_buy = initial_capital_wfv
    chained_capital_sell = initial_capital_wfv
    chained_ks_state_buy = False # Kill Switch Active state for BUY
    chained_ks_state_sell = False # Kill Switch Active state for SELL
    chained_consecutive_losses_buy = 0
    chained_consecutive_losses_sell = 0
    fold_end_timestamps: List[pd.Timestamp] = [] # To mark folds on equity curve

    for fold_idx, (train_indices, test_indices) in enumerate(tscv_wfv.split(df_m1_final_for_wfv)):
        fold_label = f"Fold_{fold_idx+1}_{fund_name_for_wfv_log}"
        wfv_logger.info(f"\n--- Processing {fold_label} (Train size: {len(train_indices)}, Test size: {len(test_indices)}) ---")

        df_train_current_fold = df_m1_final_for_wfv.iloc[train_indices].copy()
        df_test_current_fold_orig = df_m1_final_for_wfv.iloc[test_indices].copy()

        if df_test_current_fold_orig.empty: # pragma: no cover
            wfv_logger.warning(f"   Skipping {fold_label}: Test data is empty.")
            all_fold_metrics_list.append({}) # Add empty metrics for this fold to keep list length consistent
            continue
        if fold_idx == 0 and first_fold_test_data_output is None: # Save first fold test data for later SHAP
            first_fold_test_data_output = df_test_current_fold_orig.copy()
        fold_end_timestamps.append(df_test_current_fold_orig.index[-1])

        # Drift Analysis
        if drift_observer_for_wfv and not df_train_current_fold.empty: # pragma: no cover
            drift_observer_for_wfv.analyze_fold(df_train_current_fold, df_test_current_fold_orig, fold_idx, config_obj)
            drift_observer_for_wfv.export_fold_summary(output_dir_for_wfv, fold_idx) # Export per-fold drift details

        # --- Get fold-specific configuration (e.g., signal thresholds) ---
        # This might come from config_obj.entry_config_per_fold or be adjusted by drift
        fold_config_wfv = config_obj.entry_config_per_fold.get(fold_idx, {}).copy() # Start with base for this fold
        base_gain_z_thresh = fold_config_wfv.get('gain_z_thresh', getattr(config_obj, 'default_gain_z_thresh_fold', 0.3))
        if drift_observer_for_wfv: # pragma: no cover
            current_drift_summary = drift_observer_for_wfv.get_fold_drift_summary(fold_idx)
            fold_config_wfv['gain_z_thresh'] = adjust_gain_z_threshold_by_drift(
                base_gain_z_thresh, current_drift_summary, config_obj
            )
        else: # pragma: no cover
            fold_config_wfv['gain_z_thresh'] = base_gain_z_thresh # Use base if no drift observer
        wfv_logger.info(f"   Using Gain_Z Threshold for {fold_label}: {fold_config_wfv['gain_z_thresh']:.3f}")

        # <<< MODIFIED: Calculate entry signals using fold-specific config >>>
        wfv_logger.info(f"   Calculating entry signals for {fold_label} using its specific fold_config...")
        df_test_current_fold_with_signals = calculate_m1_entry_signals(
            df_m1=df_test_current_fold_orig,
            fold_specific_config=fold_config_wfv,
            strategy_config=config_obj
        )
        wfv_logger.debug(f"   Test data for {fold_label} after signal calculation: {df_test_current_fold_with_signals.shape}")
        # Now df_test_current_fold_with_signals has Entry_Long, Entry_Short, Signal_Score, Trade_Reason, Trade_Tag columns

        fold_metrics_this_run: Dict[str, Any] = {}
        for side_wfv in ["BUY", "SELL"]:
            wfv_logger.info(f"   Simulating {side_wfv} for {fold_label}...")
            # Get current capital and kill switch state for this side from previous fold
            current_capital_for_side = chained_capital_buy if side_wfv == "BUY" else chained_capital_sell
            current_ks_for_side = chained_ks_state_buy if side_wfv == "BUY" else chained_ks_state_sell
            current_losses_for_side = chained_consecutive_losses_buy if side_wfv == "BUY" else chained_consecutive_losses_sell

            # Re-initialize RiskManager's peak for the chained capital of this side
            risk_manager_obj.dd_peak = current_capital_for_side # Reset peak for this fold's sim for this side
            risk_manager_obj.soft_kill_active = False # Reset soft kill

            (df_sim_side, trade_log_side, final_equity_side, equity_history_side,
             max_dd_side, run_summary_side, blocked_log_side, model_l1_used_side,
             model_l2_used_side, ks_activated_side, cons_losses_side, ib_lot_side) = run_backtest_simulation_v34(
                df_m1_segment_pd=df_test_current_fold_with_signals, # Use data with signals calculated
                label=f"{fold_label}_{side_wfv}",
                initial_capital_segment=current_capital_for_side, # Use chained capital
                side=side_wfv,
                config_obj=config_obj,
                risk_manager_obj=risk_manager_obj, # Pass the single RM instance
                trade_manager_obj=trade_manager_obj, # Pass the single TM instance
                fund_profile=current_fund_profile_wfv,
                fold_config_override=fold_config_wfv, # Pass the adjusted fold config
                available_models=available_models_for_wfv,
                model_switcher_func=model_switcher_func_for_wfv,
                meta_min_proba_thresh_override=l1_thresh_to_use_wfv,
                current_fold_index=fold_idx,
                initial_kill_switch_state=current_ks_for_side, # Pass chained KS state
                initial_consecutive_losses=current_losses_for_side # Pass chained losses
            )

            all_fold_results_list.append(df_sim_side)
            if not trade_log_side.empty:
                all_trade_logs_list.append(trade_log_side)
            all_equity_histories_dict[f"{fold_label}_{side_wfv}"] = equity_history_side
            if blocked_log_side: # pragma: no cover
                all_blocked_logs_list.extend(blocked_log_side)
            total_ib_lot_accumulator_overall += ib_lot_side

            metrics_side_fold = calculate_metrics(
                config_obj, trade_log_side, final_equity_side, equity_history_side,
                f"{fold_label} {side_wfv}", model_l1_used_side, model_l2_used_side, run_summary_side, ib_lot_side
            )
            fold_metrics_this_run[side_wfv.lower()] = metrics_side_fold # Store BUY/SELL metrics separately

            if not trade_log_side.empty: # pragma: no cover
                export_trade_log_to_csv(trade_log_side, f"{fold_label}_{side_wfv}", output_dir_for_wfv, config_obj) # type: ignore
            export_run_summary_to_json(metrics_side_fold, f"{fold_label}_{side_wfv}_metrics", output_dir_for_wfv, config_obj) # type: ignore

            # Update chained capital and KS state for the next fold for this side
            if side_wfv == "BUY":
                chained_capital_buy = final_equity_side
                chained_ks_state_buy = ks_activated_side
                chained_consecutive_losses_buy = cons_losses_side
            else: # SELL
                chained_capital_sell = final_equity_side
                chained_ks_state_sell = ks_activated_side
                chained_consecutive_losses_sell = cons_losses_side

            wfv_logger.info(f"   Finished {side_wfv} for {fold_label}. Final Equity: ${final_equity_side:.2f}, Trades: {len(trade_log_side)}")
            gc.collect()

        all_fold_metrics_list.append(fold_metrics_this_run) # List of dicts, each dict has 'buy' and 'sell' keys
        previous_fold_metrics_data = fold_metrics_this_run # For potential future use

        del df_train_current_fold, df_test_current_fold_orig, df_test_current_fold_with_signals
        gc.collect()

    wfv_logger.info(f"\n--- Completed All {n_splits_wfv} Folds for Fund: {fund_name_for_wfv_log} ---")
    trade_log_overall = pd.DataFrame()
    if all_trade_logs_list:
        trade_log_overall = pd.concat(all_trade_logs_list, ignore_index=True)
        if "entry_time" in trade_log_overall.columns and not trade_log_overall.empty: # pragma: no cover
            trade_log_overall["entry_time"] = pd.to_datetime(trade_log_overall["entry_time"], errors='coerce')
            trade_log_overall.sort_values(by="entry_time", inplace=True)
            trade_log_overall.reset_index(drop=True, inplace=True)

    if not trade_log_overall.empty: # pragma: no cover
        export_trade_log_to_csv(trade_log_overall, f"{fund_name_for_wfv_log}_WFV_ALL_FOLDS", output_dir_for_wfv, config_obj) # type: ignore

    # Combine equity histories for overall plot
    eq_buy_hist_combined: Dict[pd.Timestamp, float] = {}
    eq_sell_hist_combined: Dict[pd.Timestamp, float] = {}
    for key_hist, hist_data_item in all_equity_histories_dict.items():
        if isinstance(key_hist, str) and f"_BUY" in key_hist: # pragma: no cover
            eq_buy_hist_combined.update(hist_data_item)
        elif isinstance(key_hist, str) and f"_SELL" in key_hist: # pragma: no cover
            eq_sell_hist_combined.update(hist_data_item)

    eq_buy_series_final = pd.Series(dict(sorted(eq_buy_hist_combined.items()))).sort_index() if eq_buy_hist_combined else pd.Series(dtype='float64', index=pd.to_datetime([]))
    if not eq_buy_series_final.empty: # pragma: no cover
        eq_buy_series_final = eq_buy_series_final[~eq_buy_series_final.index.duplicated(keep='last')]

    eq_sell_series_final = pd.Series(dict(sorted(eq_sell_hist_combined.items()))).sort_index() if eq_sell_hist_combined else pd.Series(dtype='float64', index=pd.to_datetime([]))
    if not eq_sell_series_final.empty: # pragma: no cover
        eq_sell_series_final = eq_sell_series_final[~eq_sell_series_final.index.duplicated(keep='last')]

    # Calculate overall metrics for BUY and SELL
    total_ib_lot_buy_overall = sum(
        fold_metric.get("buy", {}).get(f"Fold_{i+1}_{fund_name_for_wfv_log} BUY Total Lots Traded (IB Accumulator)", 0.0)
        for i, fold_metric in enumerate(all_fold_metrics_list) if isinstance(fold_metric.get("buy"), dict)
    )
    total_ib_lot_sell_overall = sum(
        fold_metric.get("sell", {}).get(f"Fold_{i+1}_{fund_name_for_wfv_log} SELL Total Lots Traded (IB Accumulator)", 0.0)
        for i, fold_metric in enumerate(all_fold_metrics_list) if isinstance(fold_metric.get("sell"), dict)
    )

    metrics_buy_overall = calculate_metrics(
        config_obj, trade_log_overall[trade_log_overall["side"] == "BUY"].copy() if not trade_log_overall.empty else pd.DataFrame(),
        eq_buy_series_final.iloc[-1] if not eq_buy_series_final.empty else initial_capital_wfv,
        eq_buy_series_final.to_dict() if not eq_buy_series_final.empty else {pd.Timestamp.now(tz='UTC'): initial_capital_wfv},
        f"Overall WF Buy ({fund_name_for_wfv_log})", model_type_l1_overall, model_type_l2_overall,
        {"fund_profile": current_fund_profile_wfv, "source": "run_all_folds"},
        total_ib_lot_buy_overall
    )
    metrics_sell_overall = calculate_metrics(
        config_obj, trade_log_overall[trade_log_overall["side"] == "SELL"].copy() if not trade_log_overall.empty else pd.DataFrame(),
        eq_sell_series_final.iloc[-1] if not eq_sell_series_final.empty else initial_capital_wfv,
        eq_sell_series_final.to_dict() if not eq_sell_series_final.empty else {pd.Timestamp.now(tz='UTC'): initial_capital_wfv},
        f"Overall WF Sell ({fund_name_for_wfv_log})", model_type_l1_overall, model_type_l2_overall,
        {"fund_profile": current_fund_profile_wfv, "source": "run_all_folds"},
        total_ib_lot_sell_overall
    )

    overall_summary_export = {**metrics_buy_overall, **metrics_sell_overall}
    export_run_summary_to_json(overall_summary_export, f"{fund_name_for_wfv_log}_WFV_ALL_SUMMARY", output_dir_for_wfv, config_obj)  # type: ignore

    if not eq_buy_series_final.empty: # pragma: no cover
        plot_equity_curve(config_obj, eq_buy_series_final, f"Overall Equity Curve - BUY ({fund_name_for_wfv_log})", output_dir_for_wfv, f"buy_{fund_name_for_wfv_log}_overall_wf", fold_boundaries=fold_end_timestamps)
    if not eq_sell_series_final.empty: # pragma: no cover
        plot_equity_curve(config_obj, eq_sell_series_final, f"Overall Equity Curve - SELL ({fund_name_for_wfv_log})", output_dir_for_wfv, f"sell_{fund_name_for_wfv_log}_overall_wf", fold_boundaries=fold_end_timestamps)

    # Concatenate all fold results for df_walk_forward_results
    df_walk_forward_results_final = pd.DataFrame()
    if all_fold_results_list: # pragma: no cover
        try:
            df_walk_forward_results_final = pd.concat(all_fold_results_list, axis=0, sort=False)
            if not df_walk_forward_results_final.empty:
                df_walk_forward_results_final = df_walk_forward_results_final[~df_walk_forward_results_final.index.duplicated(keep='last')] # Should already be unique due to fold separation
                df_walk_forward_results_final.sort_index(inplace=True)
        except Exception as e_concat_wfv: # pragma: no cover
            wfv_logger.error(f"Error concatenating WFV fold results: {e_concat_wfv}", exc_info=True)
            df_walk_forward_results_final = pd.DataFrame() # Ensure it's an empty DF on error

    wfv_logger.info(f"--- Finished WFV Simulation for Fund: {fund_name_for_wfv_log} ---")
    return (
        metrics_buy_overall, metrics_sell_overall,
        df_walk_forward_results_final, trade_log_overall,
        all_equity_histories_dict, all_fold_metrics_list,
        first_fold_test_data_output, # Return the test data of the first fold
        model_type_l1_overall, model_type_l2_overall, # Overall model types used
        total_ib_lot_accumulator_overall # Return total IB lots accumulated
    )

logger.info("Part 10 (Original Part 9): Walk-Forward Orchestration & Analysis Functions (Fuller Logic v4.9.18 - Corrected WFV Type Hint) Loaded and Refactored.")
# === END OF PART 10/15 ===
# ==============================================================================
# === PART 11/15: Main Execution & Pipeline Control (v4.9.13 - Refined PREPARE_TRAIN_DATA Logic) ===
# ==============================================================================
# <<< MODIFIED: Refined file saving logic in PREPARE_TRAIN_DATA mode. >>>
# <<< M1 data saved is now the feature-engineered data *before* WFV-specific signals. >>>
# <<< Trade log saved is the one generated by the WFV process. >>>
# <<< MODIFIED: [Patch v4.9.23] main - Changed fillna inplace=True to assignment to resolve FutureWarning. >>>
# <<< MODIFIED: [Patch AI Studio v4.9.1] Applied fillna assignment patch. >>>
# <<< MODIFIED: [Patch - IMPORT ERROR FIX - Step MainFunc] Added FileHandler setup in main(). >>>

import logging  # Already imported
import os  # Already imported
import sys  # Already imported
import time  # Already imported
import pandas as pd  # Already imported (or dummy if import failed)
import shutil  # For file moving in pipeline mode # Already imported
import traceback  # Already imported
from joblib import load  # For loading models # Already imported
import gc  # For memory management # Already imported
from typing import Optional, Callable, Any, Dict, List, Tuple # Ensure this is imported

# StrategyConfig, RiskManager, TradeManager, load_config_from_yaml are defined in Part 3
# DriftObserver, run_all_folds_with_threshold, calculate_metrics, plot_equity_curve are in Part 10
# train_and_export_meta_model is in Part 8
# Data processing functions (load_data, prepare_datetime, engineer_m1_features, clean_m1_data, calculate_m15_trend_zone) are in Part 5 & 6
# setup_output_directory is in Part 4
# safe_load_csv_auto is in Part 4
# simple_converter is in Part 4

# Global variables that will be SET by main() after loading config.
OUTPUT_DIR: str = "" # Will be set by main
DATA_FILE_PATH_M15: str = "" # Will be set by main
DATA_FILE_PATH_M1: str = "" # Will be set by main
M1_FEATURES_FOR_DRIFT: List[str] = [] # Will be set by main
USE_GPU_ACCELERATION: bool = True # Default, will be set by main
TRAIN_META_MODEL_BEFORE_RUN: bool = True # Default, will be set by main
config_main_obj: Optional['StrategyConfig'] = None  # type: ignore # Will be set by main
LOG_FILENAME: str = "" # Will be set by main after config load


# --- Auto-Train Trigger Function (Now uses StrategyConfig) ---
def ensure_model_files_exist(config: 'StrategyConfig', output_dir_to_check: str):  # type: ignore
    """
    Checks if required model files and their corresponding feature lists exist.
    If any are missing, it triggers the training process for those specific models.
    Uses paths and training parameters from the config object.
    """
    ensure_logger = logging.getLogger(f"{__name__}.ensure_model_files_exist")
    ensure_logger.info("\n--- (Auto-Train Check) Ensuring Model Files Exist ---")
    models_to_check = {
        'main': getattr(config, 'meta_classifier_filename', "meta_classifier.pkl"),
        'spike': getattr(config, 'spike_model_filename', "meta_classifier_spike.pkl"),
        'cluster': getattr(config, 'cluster_model_filename', "meta_classifier_cluster.pkl"),
    }
    training_needed_purposes: List[str] = []

    for model_purpose, model_filename in models_to_check.items():
        model_path = os.path.join(output_dir_to_check, model_filename)
        features_filename = f"features_{model_purpose}.json"
        features_path = os.path.join(output_dir_to_check, features_filename)
        model_exists = os.path.exists(model_path)
        features_exist = os.path.exists(features_path)
        if not model_exists or not features_exist: # pragma: no cover
            if not model_exists:
                ensure_logger.warning(f"   (Missing) Model file for '{model_purpose}' not found: {model_path}")
            if not features_exist:
                ensure_logger.warning(f"   (Missing) Features file for '{model_purpose}' not found: {features_path}")
            training_needed_purposes.append(model_purpose)
        else:
            ensure_logger.info(f"   (Found) Model and Features files for '{model_purpose}' exist.")

    if not training_needed_purposes: # pragma: no cover
        ensure_logger.info("   (Success) All required model and feature files exist. No auto-training needed.")
        return

    ensure_logger.warning(f"\n   --- Triggering Auto-Training for Missing Models: {training_needed_purposes} ---") # pragma: no cover
    ensure_logger.info("      Loading base data for training (paths from config)...") # pragma: no cover
    trade_log_df_base_auto_train: Optional[pd.DataFrame] = None # pragma: no cover
    m1_data_path_auto_train: Optional[str] = None # pragma: no cover

    base_log_path_cfg = getattr(config, 'base_train_trade_log_path', os.path.join(output_dir_to_check, "trade_log_v32_walkforward")) # pragma: no cover
    base_m1_path_cfg = getattr(config, 'base_train_m1_data_path', os.path.join(output_dir_to_check, "final_data_m1_v32_walkforward")) # pragma: no cover

    try: # pragma: no cover
        log_path_gz = base_log_path_cfg + ".csv.gz"
        log_path_csv = base_log_path_cfg + ".csv"
        resolved_log_path: Optional[str] = None
        if os.path.exists(log_path_gz):
            resolved_log_path = log_path_gz
        elif os.path.exists(log_path_csv):
            resolved_log_path = log_path_csv
        else:
            # Fallback to prep_data suffixed files if primary base files not found
            default_fund_prep = getattr(config, 'default_fund_name_for_prep_fallback', "PREP_DEFAULT")
            fallback_log_gz = os.path.join(output_dir_to_check, f"trade_log_v32_walkforward_prep_data_{default_fund_prep}.csv.gz")
            fallback_log_csv = os.path.join(output_dir_to_check, f"trade_log_v32_walkforward_prep_data_{default_fund_prep}.csv")
            if os.path.exists(fallback_log_gz):
                resolved_log_path = fallback_log_gz
                ensure_logger.info(f"      [Fallback Log] Using: {os.path.basename(resolved_log_path)}")
            elif os.path.exists(fallback_log_csv):
                resolved_log_path = fallback_log_csv
                ensure_logger.info(f"      [Fallback Log] Using: {os.path.basename(resolved_log_path)}")
            else:
                raise FileNotFoundError(f"Base trade log for auto-train not found. Checked primary and fallback for '{default_fund_prep}'.")
        ensure_logger.info(f"      Using Trade Log for Auto-Training: {os.path.basename(resolved_log_path)}")
        trade_log_df_base_auto_train = safe_load_csv_auto(resolved_log_path)  # type: ignore
        if trade_log_df_base_auto_train is None or trade_log_df_base_auto_train.empty:
            ensure_logger.error("      Loaded trade log for auto-training is empty or None. Skipping training.")
            return
        # Ensure datetime columns are parsed
        time_cols_auto_train = ["entry_time", "close_time", "BE_Triggered_Time"] # Add others if present
        for col_auto_train in time_cols_auto_train:
            if col_auto_train in trade_log_df_base_auto_train.columns:
                trade_log_df_base_auto_train[col_auto_train] = pd.to_datetime(trade_log_df_base_auto_train[col_auto_train], errors='coerce')
        # Ensure context columns exist for filtering (spike_score, cluster)
        for ctx_col in ['cluster', 'spike_score']: # Ensure context columns exist for filtering
            if ctx_col not in trade_log_df_base_auto_train.columns:
                trade_log_df_base_auto_train[ctx_col] = 0.0 if ctx_col == 'spike_score' else 0
                ensure_logger.info(f"      Added missing context column '{ctx_col}' to auto-train log with default values.")

        m1_path_gz = base_m1_path_cfg + ".csv.gz"
        m1_path_csv = base_m1_path_cfg + ".csv"
        if os.path.exists(m1_path_gz):
            m1_data_path_auto_train = m1_path_gz
        elif os.path.exists(m1_path_csv):
            m1_data_path_auto_train = m1_path_csv
        else:
            default_fund_prep_m1 = getattr(config, 'default_fund_name_for_prep_fallback', "PREP_DEFAULT")
            fallback_m1_gz = os.path.join(output_dir_to_check, f"final_data_m1_v32_walkforward_prep_data_{default_fund_prep_m1}.csv.gz")
            fallback_m1_csv = os.path.join(output_dir_to_check, f"final_data_m1_v32_walkforward_prep_data_{default_fund_prep_m1}.csv")
            if os.path.exists(fallback_m1_gz):
                m1_data_path_auto_train = fallback_m1_gz
                ensure_logger.info(f"      [Fallback M1] Using: {os.path.basename(m1_data_path_auto_train)}")
            elif os.path.exists(fallback_m1_csv):
                m1_data_path_auto_train = fallback_m1_csv
                ensure_logger.info(f"      [Fallback M1] Using: {os.path.basename(m1_data_path_auto_train)}")
            else:
                raise FileNotFoundError(f"Base M1 data for auto-train not found. Checked primary and fallback for '{default_fund_prep_m1}'.")
        ensure_logger.info(f"      Using M1 Data Path for Auto-Training: {os.path.basename(m1_data_path_auto_train)}")

    except FileNotFoundError as fnf_auto_train: # pragma: no cover
        ensure_logger.critical(f"      (Error) Required data file not found for auto-train: {fnf_auto_train}")
        ensure_logger.critical("         Skipping auto-training due to missing data.")
        return
    except Exception as e_load_auto_train: # pragma: no cover
        ensure_logger.error(f"      (Error) Failed to load/process base data for auto-training: {e_load_auto_train}", exc_info=True)
        ensure_logger.error("         Skipping auto-training.")
        return

    for model_purpose_train_auto in training_needed_purposes: # pragma: no cover
        ensure_logger.info(f"\n      --- Auto-Training Model: {model_purpose_train_auto.upper()} ---")
        trade_log_filtered_auto_train: Optional[pd.DataFrame] = None
        if model_purpose_train_auto == 'main':
            trade_log_filtered_auto_train = trade_log_df_base_auto_train.copy() # type: ignore
        elif model_purpose_train_auto == 'spike':
            spike_filter_thresh_auto = getattr(config, 'auto_train_spike_filter_threshold', 0.6)
            if 'spike_score' in trade_log_df_base_auto_train.columns: # type: ignore
                trade_log_filtered_auto_train = trade_log_df_base_auto_train[trade_log_df_base_auto_train['spike_score'] > spike_filter_thresh_auto].copy() # type: ignore
            else:
                ensure_logger.warning(" 'spike_score' not in log for spike model auto-train. Using full log.")
                trade_log_filtered_auto_train = trade_log_df_base_auto_train.copy() # type: ignore
        elif model_purpose_train_auto == 'cluster':
            cluster_filter_val_auto = getattr(config, 'auto_train_cluster_filter_value', 2)
            if 'cluster' in trade_log_df_base_auto_train.columns: # type: ignore
                trade_log_filtered_auto_train = trade_log_df_base_auto_train[trade_log_df_base_auto_train['cluster'] == cluster_filter_val_auto].copy() # type: ignore
            else:
                ensure_logger.warning(" 'cluster' not in log for cluster model auto-train. Using full log.")
                trade_log_filtered_auto_train = trade_log_df_base_auto_train.copy() # type: ignore
        else:
            ensure_logger.warning(f"         Unknown model purpose '{model_purpose_train_auto}' for auto-train filtering. Using full log.")
            trade_log_filtered_auto_train = trade_log_df_base_auto_train.copy() # type: ignore

        if trade_log_filtered_auto_train is None or trade_log_filtered_auto_train.empty:
            ensure_logger.warning(f"         No data available after filtering for '{model_purpose_train_auto}' model. Skipping training.")
            continue
        try:
            saved_paths_auto_train, _ = train_and_export_meta_model(  # type: ignore
                config=config,
                output_dir=output_dir_to_check,
                model_purpose=model_purpose_train_auto,
                trade_log_df_override=trade_log_filtered_auto_train,
                m1_data_path=m1_data_path_auto_train,
                enable_optuna_tuning_override=getattr(config, 'auto_train_enable_optuna', False),
                enable_dynamic_feature_selection_override=getattr(config, 'auto_train_enable_dynamic_features', True)
            )
            if not saved_paths_auto_train or model_purpose_train_auto not in saved_paths_auto_train:
                ensure_logger.error(f"         (Error) Auto-training for '{model_purpose_train_auto}' did not save the model as expected.")
            else:
                ensure_logger.info(f"         (Success) Auto-training for '{model_purpose_train_auto}' completed and saved: {saved_paths_auto_train[model_purpose_train_auto]}")
        except NameError as ne_auto_train_call: # Check if train_and_export_meta_model is defined
            ensure_logger.critical(f"      (CRITICAL) NameError during auto-training call for '{model_purpose_train_auto}': {ne_auto_train_call}. Check function definitions.", exc_info=True)
            break # Stop further auto-training attempts if a core function is missing
        except Exception as e_auto_train_run:
            ensure_logger.error(f"         (Error) Exception during auto-training for '{model_purpose_train_auto}': {e_auto_train_run}", exc_info=True)
        finally:
            if trade_log_filtered_auto_train is not None:
                del trade_log_filtered_auto_train
                gc.collect()
    if trade_log_df_base_auto_train is not None: # pragma: no cover
        del trade_log_df_base_auto_train
        gc.collect()
    ensure_logger.info("--- (Auto-Train Check) Finished ---")


# --- Main Execution Function (Refactored to use StrategyConfig) ---
def main(run_mode: str = 'FULL_PIPELINE', config_file: str = "config.yaml", suffix_from_prev_step: Optional[str] = None) -> Optional[str]:
    main_exec_logger_func = logging.getLogger(f"{__name__}.main")
    start_time_main_call = time.time()
    main_exec_logger_func.info(f"\n(Starting) Gold AI Main Execution (Mode: {run_mode}, Config: {config_file})...")
    current_run_suffix_for_main: Optional[str] = ""

    global config_main_obj
    config_main_obj = load_config_from_yaml(config_file)
    if config_main_obj is None: # pragma: no cover
        main_exec_logger_func.critical("CRITICAL: Failed to load StrategyConfig. Exiting.")
        return None
    main_exec_logger_func.info(f"StrategyConfig loaded. Initial Capital: ${config_main_obj.initial_capital:.2f}, Risk/Trade: {config_main_obj.risk_per_trade:.3f}")

    global OUTPUT_DIR, LOG_FILENAME # Ensure global OUTPUT_DIR and LOG_FILENAME are updated
    OUTPUT_DIR = setup_output_directory(config_main_obj.output_base_dir, config_main_obj.output_dir_name)
    # [Patch - IMPORT ERROR FIX - Step MainFunc] Configure FileHandler here
    LOG_FILENAME = f"gold_ai_v{MINIMAL_SCRIPT_VERSION.split('_')[0]}_{config_main_obj.output_dir_name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    log_file_path_main = os.path.join(OUTPUT_DIR, LOG_FILENAME)
    
    # Remove existing file handlers if any, then add the new one
    # This ensures that if main() is called multiple times (e.g., in pipeline), logs go to the correct new file.
    for handler_main in logger.handlers[:]: # Iterate over a copy
        if isinstance(handler_main, logging.FileHandler):
            logger.removeHandler(handler_main)
            handler_main.close()
            main_exec_logger_func.debug(f"Removed existing FileHandler: {handler_main.baseFilename}")
            
    try:
        fh_main = logging.FileHandler(log_file_path_main, mode='a', encoding='utf-8')
        fh_main.setLevel(logging.DEBUG) # File handler logs everything (DEBUG and above)
        fh_formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(filename)s:%(lineno)d - %(message)s')
        fh_main.setFormatter(fh_formatter_main)
        logger.addHandler(fh_main)
        main_exec_logger_func.info(f"[Patch - IMPORT ERROR FIX - Step MainFunc] FileHandler configured. Logging to: {log_file_path_main}")
    except Exception as e_fh_main: # pragma: no cover
        main_exec_logger_func.error(f"CRITICAL: Failed to set up FileHandler for logging at {log_file_path_main}: {e_fh_main}", exc_info=True)
        # Continue with StreamHandler only if FileHandler fails

    main_exec_logger_func.info(f"Global OUTPUT_DIR has been set to: {OUTPUT_DIR}")
    main_exec_logger_func.info(f"Global LOG_FILENAME has been set to: {LOG_FILENAME}")


    # Initialize RiskManager and TradeManager with the loaded config
    risk_manager_for_main = RiskManager(config_main_obj)
    trade_manager_for_main = TradeManager(config_main_obj, risk_manager_for_main)

    # Update other relevant globals from config
    global USE_GPU_ACCELERATION, TRAIN_META_MODEL_BEFORE_RUN, DATA_FILE_PATH_M15, DATA_FILE_PATH_M1
    USE_GPU_ACCELERATION = config_main_obj.use_gpu_acceleration # This might be re-evaluated by setup_gpu_acceleration if called
    TRAIN_META_MODEL_BEFORE_RUN = config_main_obj.train_meta_model_before_run
    DATA_FILE_PATH_M15 = config_main_obj.data_file_path_m15
    DATA_FILE_PATH_M1 = config_main_obj.data_file_path_m1
    main_exec_logger_func.info(f"  Configured USE_GPU_ACCELERATION (initial from config): {USE_GPU_ACCELERATION}")
    main_exec_logger_func.info(f"  Configured TRAIN_META_MODEL_BEFORE_RUN: {TRAIN_META_MODEL_BEFORE_RUN}")
    main_exec_logger_func.info(f"  Data M15 Path: {DATA_FILE_PATH_M15}")
    main_exec_logger_func.info(f"  Data M1 Path: {DATA_FILE_PATH_M1}")

    try:
        if 'setup_fonts' in globals() and callable(setup_fonts): # pragma: no cover
            setup_fonts(OUTPUT_DIR)
        # GPU utilization will be printed after setup_gpu_acceleration is called in __main__
    except Exception as e_setup_main_call_in_main: # pragma: no cover
        main_exec_logger_func.warning(f"(Warning) Error during post-config setup (fonts): {e_setup_main_call_in_main}")

    train_model_in_main = False
    run_final_backtest_in_main = False
    train_model_flag_from_config_main = config_main_obj.train_meta_model_before_run

    main_exec_logger_func.info(f"   Run Mode Selected: {run_mode}")
    if run_mode == 'PREPARE_TRAIN_DATA':
        run_final_backtest_in_main = True # WFV is run to generate the log
    elif run_mode == 'TRAIN_MODEL_ONLY':
        train_model_in_main = True
    elif run_mode == 'FULL_RUN':
        run_final_backtest_in_main = True
        train_model_in_main = train_model_flag_from_config_main # Respect config for FULL_RUN
    elif run_mode == 'FULL_PIPELINE': # pragma: no cover
        main_exec_logger_func.info("\n(Pipeline) FULL PIPELINE execution started...")
        main_exec_logger_func.info("\n--- Pipeline Step 1: PREPARE_TRAIN_DATA mode ---")
        prep_suffix_pipeline_run = main(run_mode='PREPARE_TRAIN_DATA', config_file=config_file)
        if prep_suffix_pipeline_run is None:
            main_exec_logger_func.critical("Pipeline Step 1 (PREPARE_TRAIN_DATA) failed.")
            return None

        main_exec_logger_func.info("\n--- Pipeline Step 2: Renaming/Moving generated files ---")
        log_file_generated_pipeline = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{prep_suffix_pipeline_run}.csv.gz")
        data_file_generated_pipeline = os.path.join(OUTPUT_DIR, f"final_data_m1_v32_walkforward{prep_suffix_pipeline_run}.csv.gz")
        # Target paths should ideally come from config for consistency
        target_log_path_pipeline = getattr(config_main_obj, 'base_train_trade_log_path', os.path.join(OUTPUT_DIR, "trade_log_v32_walkforward")) + ".csv.gz"
        target_m1_path_pipeline = getattr(config_main_obj, 'base_train_m1_data_path', os.path.join(OUTPUT_DIR, "final_data_m1_v32_walkforward")) + ".csv.gz"
        rename_failed_pipeline_run = False
        try:
            if os.path.exists(log_file_generated_pipeline) and os.path.exists(data_file_generated_pipeline):
                if os.path.exists(target_log_path_pipeline): # Remove old target if it exists
                    main_exec_logger_func.info(f"   Removing existing target log: {target_log_path_pipeline}")
                    os.remove(target_log_path_pipeline)
                shutil.move(log_file_generated_pipeline, target_log_path_pipeline)
                main_exec_logger_func.info(f"   Moved {os.path.basename(log_file_generated_pipeline)} to {os.path.basename(target_log_path_pipeline)}")

                if os.path.exists(target_m1_path_pipeline): # Remove old target if it exists
                    main_exec_logger_func.info(f"   Removing existing target M1 data: {target_m1_path_pipeline}")
                    os.remove(target_m1_path_pipeline)
                shutil.move(data_file_generated_pipeline, target_m1_path_pipeline)
                main_exec_logger_func.info(f"   Moved {os.path.basename(data_file_generated_pipeline)} to {os.path.basename(target_m1_path_pipeline)}")
            else:
                main_exec_logger_func.error(f"   Generated files for pipeline rename not found. Checked: '{log_file_generated_pipeline}', '{data_file_generated_pipeline}'")
                rename_failed_pipeline_run = True
        except Exception as e_rename_pipeline_run:
            main_exec_logger_func.error(f"   Error renaming files in pipeline: {e_rename_pipeline_run}", exc_info=True)
            rename_failed_pipeline_run = True
        if rename_failed_pipeline_run:
            main_exec_logger_func.critical("Pipeline stopped due to file rename failure.")
            return None

        main_exec_logger_func.info("\n--- Pipeline Step 3: Ensure Models Exist (Auto-Train) ---")
        ensure_model_files_exist(config_main_obj, OUTPUT_DIR) # Call with the config

        main_exec_logger_func.info("\n--- Pipeline Step 4: FULL_RUN mode ---")
        full_run_suffix_pipeline_exec_main = main(run_mode='FULL_RUN', config_file=config_file) # Suffix from this run will be the final one
        main_exec_logger_func.info("\n(Pipeline) FULL PIPELINE finished.")
        return full_run_suffix_pipeline_exec_main
    else: # pragma: no cover
        main_exec_logger_func.warning(f"(Warning) Unknown run_mode '{run_mode}'. Defaulting to FULL_RUN.")
        run_mode = 'FULL_RUN'
        run_final_backtest_in_main = True
        train_model_in_main = train_model_flag_from_config_main

    df_m1_final_for_main_exec: Optional[pd.DataFrame] = None
    global M1_FEATURES_FOR_DRIFT
    M1_FEATURES_FOR_DRIFT = [] # Initialize / Reset
    df_m15_loaded_for_main: Optional[pd.DataFrame] = None # Store the prepared M15 data

    if run_mode in ['FULL_RUN', 'PREPARE_TRAIN_DATA']:
        main_exec_logger_func.info(f"\n--- ({run_mode}) Starting Data Preparation ---")
        try:
            main_exec_logger_func.info(f"   Loading M15 data from: {config_main_obj.data_file_path_m15}")
            df_m15_raw = load_data(config_main_obj.data_file_path_m15, "M15_Main")
            if df_m15_raw is None or df_m15_raw.empty: # pragma: no cover
                main_exec_logger_func.critical("M15 data loading failed or returned empty. Cannot proceed.")
                return None
            df_m15_loaded_for_main = prepare_datetime(df_m15_raw, "M15_Main", config=config_main_obj) # Pass config
            if df_m15_loaded_for_main is None or df_m15_loaded_for_main.empty: # pragma: no cover
                main_exec_logger_func.critical("M15 data preparation resulted in empty or None DataFrame. Cannot proceed.")
                if df_m15_raw is not None: del df_m15_raw; gc.collect()
                return None
            df_m15_trend_zone = calculate_m15_trend_zone(df_m15_loaded_for_main, config_main_obj) # Pass config
            del df_m15_raw, df_m15_loaded_for_main # Free memory
            gc.collect()
            main_exec_logger_func.info(f"   M15 data prepared and Trend Zone calculated. Trend Zone Shape: {df_m15_trend_zone.shape if df_m15_trend_zone is not None else 'N/A'}")

            main_exec_logger_func.info(f"   Loading M1 data from: {config_main_obj.data_file_path_m1}")
            df_m1_raw = load_data(config_main_obj.data_file_path_m1, "M1_Main")
            if df_m1_raw is None or df_m1_raw.empty: # pragma: no cover
                main_exec_logger_func.critical("M1 data loading failed or returned empty. Cannot proceed.")
                return None
            df_m1_prepared = prepare_datetime(df_m1_raw, "M1_Main", config=config_main_obj) # Pass config
            if df_m1_prepared is None or df_m1_prepared.empty: # pragma: no cover
                main_exec_logger_func.critical("M1 data preparation resulted in empty or None DataFrame. Cannot proceed.")
                if df_m1_raw is not None: del df_m1_raw; gc.collect()
                return None
            del df_m1_raw # Free memory
            gc.collect()
            main_exec_logger_func.info(f"   M1 data prepared. Shape: {df_m1_prepared.shape}")

            main_exec_logger_func.info("   Merging M15 Trend Zone into M1 data...")
            if df_m15_trend_zone is None or df_m15_trend_zone.empty: # pragma: no cover
                main_exec_logger_func.warning("   M15 Trend Zone is empty. M1 data will have 'NEUTRAL' Trend_Zone.")
                df_m1_with_trend = df_m1_prepared.copy()
                df_m1_with_trend['Trend_Zone'] = "NEUTRAL"
            else:
                df_m1_with_trend = pd.merge_asof(
                    df_m1_prepared.sort_index(),
                    df_m15_trend_zone.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction='backward',
                    tolerance=pd.Timedelta(minutes=config_main_obj.m15_trend_merge_tolerance_minutes)
                )
            main_exec_logger_func.debug("   [Patch AI Studio v4.9.1] Applying fillna for Trend_Zone without inplace=True.")
            df_m1_with_trend['Trend_Zone'] = df_m1_with_trend['Trend_Zone'].fillna("NEUTRAL")
            df_m1_with_trend['Trend_Zone'] = df_m1_with_trend['Trend_Zone'].astype('category')
            main_exec_logger_func.info(f"   M1 data after Trend Zone merge: {df_m1_with_trend.shape}")
            del df_m1_prepared
            if df_m15_trend_zone is not None: del df_m15_trend_zone
            gc.collect()

            main_exec_logger_func.info("   Engineering M1 features...")
            lag_features_setting = config_main_obj.lag_features_config # Pass specific lag config if available
            df_m1_engineered = engineer_m1_features(df_m1_with_trend, config_main_obj, lag_features_setting) # Pass config
            main_exec_logger_func.info(f"   M1 data after feature engineering: {df_m1_engineered.shape}")
            del df_m1_with_trend
            gc.collect()

            main_exec_logger_func.info("   Cleaning M1 data...")
            df_m1_cleaned, m1_features_drift_list = clean_m1_data(df_m1_engineered, config_main_obj) # Pass config
            M1_FEATURES_FOR_DRIFT = m1_features_drift_list # Set global for drift observer
            main_exec_logger_func.info(f"   M1 data after cleaning: {df_m1_cleaned.shape}. Features for drift: {len(M1_FEATURES_FOR_DRIFT)}")
            del df_m1_engineered
            gc.collect()

            df_m1_final_for_main_exec = df_m1_cleaned.copy() # This is the fully prepared M1 data
            del df_m1_cleaned
            gc.collect()

            main_exec_logger_func.info(f"   (Success) Data Preparation complete. Final M1 (features only) shape for WFV: {df_m1_final_for_main_exec.shape}")

        except Exception as e_data_prep_main: # pragma: no cover
            main_exec_logger_func.critical(f"CRITICAL Error during Data Preparation in main(): {e_data_prep_main}", exc_info=True)
            return None

    if train_model_in_main and run_mode != 'PREPARE_TRAIN_DATA': # pragma: no cover
        main_exec_logger_func.info(f"\n--- ({run_mode}) Ensuring models are trained/exist (called from main) ---")
        ensure_model_files_exist(config_main_obj, OUTPUT_DIR) # Pass the config
        current_run_suffix_for_main = "_train_models_completed_in_main"
        if not run_final_backtest_in_main: # If only training, exit after this
            main_exec_logger_func.info("Model training/check complete for TRAIN_MODEL_ONLY mode (called from main).")
            return current_run_suffix_for_main

    # Load models for FULL_RUN mode
    available_models_for_main_run: Dict[str, Any] = {}
    if run_mode == 'FULL_RUN': # pragma: no cover
        main_exec_logger_func.info("\n--- Loading Models and Features for FULL_RUN (from main) ---")
        models_to_load_main_run = {
            'main': config_main_obj.meta_classifier_filename,
            'spike': config_main_obj.spike_model_filename,
            'cluster': config_main_obj.cluster_model_filename,
        }
        all_models_loaded_successfully = True
        for model_purpose_load_main, model_filename_load_main in models_to_load_main_run.items():
            model_path_load_main = os.path.join(OUTPUT_DIR, model_filename_load_main)
            features_list_load_main = load_features_for_model(model_purpose_load_main, OUTPUT_DIR) # Uses output_dir
            if os.path.exists(model_path_load_main) and features_list_load_main:
                try:
                    loaded_model_main = load(model_path_load_main) # joblib.load
                    available_models_for_main_run[model_purpose_load_main] = {
                        "model": loaded_model_main,
                        "features": features_list_load_main
                    }
                    main_exec_logger_func.info(f"   Successfully loaded model '{model_purpose_load_main}' and its {len(features_list_load_main)} features.")
                except Exception as e_load_model_main:
                    main_exec_logger_func.error(f"   Failed to load model '{model_purpose_load_main}' from {model_path_load_main}: {e_load_model_main}")
                    if model_purpose_load_main == 'main': # Critical if main model fails
                        all_models_loaded_successfully = False
            else:
                main_exec_logger_func.warning(f"   Model or features file for '{model_purpose_load_main}' not found. Path: {model_path_load_main}, Features: {'Found' if features_list_load_main else 'Not Found'}")
                if model_purpose_load_main == 'main': # Critical if main model is missing
                    all_models_loaded_successfully = False

        if not all_models_loaded_successfully:
            main_exec_logger_func.critical("CRITICAL: Main model or its features could not be loaded for FULL_RUN. Cannot proceed.")
            return None
        main_exec_logger_func.info(f"   All necessary models for FULL_RUN loaded: {list(available_models_for_main_run.keys())}")

    # Initialize DriftObserver if needed
    drift_observer_for_main_run: Optional[DriftObserver] = None
    if df_m1_final_for_main_exec is not None and run_mode != 'TRAIN_MODEL_ONLY': # pragma: no cover
        m1_features_for_drift_main_run = config_main_obj.m1_features_for_drift if config_main_obj.m1_features_for_drift is not None else M1_FEATURES_FOR_DRIFT
        if m1_features_for_drift_main_run:
            try:
                drift_observer_for_main_run = DriftObserver(m1_features_for_drift_main_run)
            except NameError: # Should not happen if DriftObserver is defined
                main_exec_logger_func.warning("Class 'DriftObserver' not found.")
                drift_observer_for_main_run = None
        else:
            main_exec_logger_func.warning("(Warning) No M1 features for drift observation.")

    # Run Walk-Forward Simulation
    if run_final_backtest_in_main and df_m1_final_for_main_exec is not None:
        main_exec_logger_func.info(f"\n--- Starting Walk-Forward Simulation for {run_mode} (from main) ---")
        l1_threshold_main_run_val = config_main_obj.meta_min_proba_thresh

        # Determine funds to run
        funds_to_run_loop_main_exec: Dict[str, Any] = {}
        multi_fund_config_main_exec = config_main_obj.multi_fund_mode
        fund_profiles_from_config_main_exec = config_main_obj.fund_profiles

        if multi_fund_config_main_exec and fund_profiles_from_config_main_exec: # pragma: no cover
            funds_to_run_loop_main_exec = fund_profiles_from_config_main_exec
        else:
            default_fund_name_main_exec_loop = config_main_obj.default_fund_name
            default_profile_main_exec_loop = fund_profiles_from_config_main_exec.get(default_fund_name_main_exec_loop, {"risk": config_main_obj.risk_per_trade, "mm_mode": "balanced"})
            funds_to_run_loop_main_exec = {default_fund_name_main_exec_loop: default_profile_main_exec_loop}

        overall_run_suffix_parts_main_exec: List[str] = []
        trade_log_overall_for_prep_data_save = pd.DataFrame() # For PREPARE_TRAIN_DATA mode

        for fund_name_main_exec_run, fund_profile_main_exec_run_loop in funds_to_run_loop_main_exec.items():
            config_main_obj.current_fund_name_for_logging = fund_name_main_exec_run # Set for logging context
            main_exec_logger_func.info("\n" + "=" * 20 + f" STARTING FUND (main exec): {fund_name_main_exec_run} " + "=" * 20)
            try:
                (metrics_buy_overall_main_exec, metrics_sell_overall_main_exec, df_walk_forward_results_pd_main_exec, trade_log_wf_main_exec,
                 all_equity_histories_main_exec, all_fold_metrics_main_exec, first_fold_test_data_shap_main_exec,
                 model_type_l1_sim_main_exec, model_type_l2_sim_main_exec, total_ib_lot_fund_main_exec) = run_all_folds_with_threshold(
                    config_obj=config_main_obj, risk_manager_obj=risk_manager_for_main, trade_manager_obj=trade_manager_for_main,
                    df_m1_final_for_wfv=df_m1_final_for_main_exec,
                    output_dir_for_wfv=OUTPUT_DIR,
                    available_models_for_wfv=available_models_for_main_run if run_mode == 'FULL_RUN' else None,
                    model_switcher_func_for_wfv=select_model_for_trade if run_mode == 'FULL_RUN' else None, # Pass the function itself
                    drift_observer_for_wfv=drift_observer_for_main_run,
                    current_l1_threshold_override_for_wfv=l1_threshold_main_run_val,
                    fund_profile_for_wfv=fund_profile_main_exec_run_loop
                )

                fund_suffix_main_exec_loop = f"_{fund_name_main_exec_run}"
                if run_mode == 'PREPARE_TRAIN_DATA':
                    if trade_log_wf_main_exec is not None and not trade_log_wf_main_exec.empty:
                        trade_log_overall_for_prep_data_save = trade_log_wf_main_exec.copy() # <<< MODIFIED: Use this for saving
                    else: # pragma: no cover
                        main_exec_logger_func.warning(f"No trade log returned from WFV for PREPARE_TRAIN_DATA fund: {fund_name_main_exec_run}")

                overall_run_suffix_parts_main_exec.append(fund_suffix_main_exec_loop)

            except Exception as e_main_fund_loop_exec: # pragma: no cover
                main_exec_logger_func.critical(f"   (CRITICAL) Error during WFV for fund '{fund_name_main_exec_run}': {e_main_fund_loop_exec}", exc_info=True)
                if not multi_fund_config_main_exec: # If single fund mode and it fails, exit
                    return None
                else: # If multi-fund, log and continue to next fund
                    continue # To the next fund in the loop
        current_run_suffix_for_main = "".join(overall_run_suffix_parts_main_exec) if overall_run_suffix_parts_main_exec else "_run_completed_main_exec"

        # <<< MODIFIED: Refined file saving logic in PREPARE_TRAIN_DATA mode. >>>
        if run_mode == 'PREPARE_TRAIN_DATA':
            prep_data_fund_name_final_save = config_main_obj.default_fund_name_for_prep_fallback # Use specific suffix for prep data
            current_run_suffix_for_main = f"_prep_data_{prep_data_fund_name_final_save}"

            if df_m1_final_for_main_exec is not None: # This is the feature-engineered M1 data
                m1_save_path_prep_final = os.path.join(OUTPUT_DIR, f"final_data_m1_v32_walkforward{current_run_suffix_for_main}.csv.gz")
                df_m1_final_for_main_exec.to_csv(m1_save_path_prep_final, index=True, encoding="utf-8", compression="gzip")
                main_exec_logger_func.info(f"   (PREP_DATA) Saved final M1 data (features only): {os.path.basename(m1_save_path_prep_final)}")

            if not trade_log_overall_for_prep_data_save.empty: # This is the log from WFV run in prep mode
                log_save_path_prep_final = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{current_run_suffix_for_main}.csv.gz")
                trade_log_overall_for_prep_data_save.to_csv(log_save_path_prep_final, index=False, encoding="utf-8", compression="gzip")
                main_exec_logger_func.info(f"   (PREP_DATA) Saved generated trade log from WFV: {os.path.basename(log_save_path_prep_final)}")
            else: # pragma: no cover
                main_exec_logger_func.warning("   (PREP_DATA) No trade log generated/returned from WFV to save for PREPARE_TRAIN_DATA mode.")
            return current_run_suffix_for_main # Exit after PREPARE_TRAIN_DATA

    # Shutdown pynvml if used (This should ideally be in the __main__ block's finally)
    # For now, keeping it here as per original structure, but it's better in __main__
    # global pynvml, nvml_handle # Already global
    # if pynvml and nvml_handle: # pragma: no cover
    #     try:
    #         pynvml.nvmlShutdown() # type: ignore
    #         main_exec_logger_func.info("   (Info) pynvml shutdown (from main function).")
    #     except Exception as e_nvml_shutdown:
    #         main_exec_logger_func.warning(f"   (Warning) Error shutting down pynvml (from main function): {e_nvml_shutdown}")

    end_time_main_exec_func = time.time()
    main_exec_logger_func.info(f"\n--- Main function (Mode: {run_mode}, Config: {config_file}) finished in {end_time_main_exec_func - start_time_main_call:.2f} sec ---")
    return current_run_suffix_for_main

logger.info("Part 11 (Original Part 10): Main Execution & Pipeline Control Loaded and Refactored with Full Data/Model Logic.")
# === END OF PART 11/15 ===
# === START OF PART 12/15 ===
# ==============================================================================
# === PART 12: MT5 Connector (Placeholder) (v4.9.0 - Enterprise Refactor) ===
# ==============================================================================
# <<< MODIFIED: Enterprise Refactor - Loggers made more specific. >>>
# <<< MT5 parameters here are illustrative; real credentials should be in secure config. >>>

import logging # Already imported
import time # Already imported
# import MetaTrader5 as mt5 # Import commented out as it's a placeholder
# import pandas as pd # Not used in placeholder, can be removed if not needed by actual implementation

mt5_logger = logging.getLogger(f"{__name__}.MT5Connector") # Logger for this module/part
mt5_logger.info("Loading Part 12: MT5 Connector (Placeholder)...")

# --- MT5 Connection Parameters (Illustrative Placeholders) ---
# In a real application, these should be loaded from a secure configuration system
# (e.g., environment variables, encrypted config, or StrategyConfig if appropriate for non-sensitive parts)
MT5_LOGIN_DEFAULT = 12345678
MT5_PASSWORD_DEFAULT = "YOUR_MT5_PASSWORD_HERE" # Sensitive, should not be hardcoded
MT5_SERVER_DEFAULT = "YOUR_MT5_SERVER_HERE"
MT5_PATH_DEFAULT = "C:\\Program Files\\MetaTrader 5\\terminal64.exe" # Example path

# --- Placeholder Functions ---

def initialize_mt5(config: 'StrategyConfig | None' = None) -> bool: # type: ignore
    """
    Placeholder function to initialize connection to MetaTrader 5 terminal.
    In a real implementation, would use connection details from `config`.
    (Currently does nothing but log).
    """
    init_logger = logging.getLogger(f"{__name__}.MT5Connector.initialize_mt5") # Specific logger
    init_logger.info("Attempting to initialize MT5 connection (Placeholder)...")

    login = getattr(config, 'mt5_login', MT5_LOGIN_DEFAULT) if config else MT5_LOGIN_DEFAULT
    password = getattr(config, 'mt5_password', MT5_PASSWORD_DEFAULT) if config else MT5_PASSWORD_DEFAULT # Sensitive!
    server = getattr(config, 'mt5_server', MT5_SERVER_DEFAULT) if config else MT5_SERVER_DEFAULT
    path = getattr(config, 'mt5_path', MT5_PATH_DEFAULT) if config else MT5_PATH_DEFAULT

    init_logger.debug(f"  Using MT5 params - Login: {login}, Server: {server}, Path: {path}")
    if password == MT5_PASSWORD_DEFAULT and password != "YOUR_MT5_PASSWORD_HERE": # Basic check if default was changed
        init_logger.warning("  Using default MT5 password placeholder. Ensure this is configured securely for live trading.")
    elif password == "YOUR_MT5_PASSWORD_HERE":
         init_logger.warning("  MT5 Password is set to the default placeholder 'YOUR_MT5_PASSWORD_HERE'. Connection will likely fail.")


    # Example connection logic (commented out):
    # global mt5 # Ensure mt5 is accessible if uncommented
    # if not mt5.initialize(path=path, login=login, password=password, server=server):
    #     init_logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
    #     if hasattr(mt5, 'shutdown'): mt5.shutdown()
    #     return False
    # else:
    #     init_logger.info("MT5 initialized successfully (Placeholder).")
    #     acc_info = mt5.account_info()
    #     if acc_info:
    #         init_logger.info(f"Connected to MT5 Account: {acc_info.login} on {acc_info.server}")
    #     else:
    #         init_logger.warning("Could not retrieve MT5 account info (Placeholder).")
    #     return True
    init_logger.warning("   MT5 connection logic is currently commented out (Placeholder). Returning False.")
    return False

def shutdown_mt5():
    """
    Placeholder function to shut down the MetaTrader 5 connection.
    (Currently does nothing but log).
    """
    sd_logger = logging.getLogger(f"{__name__}.MT5Connector.shutdown_mt5") # Specific logger
    sd_logger.info("Attempting to shut down MT5 connection (Placeholder)...")
    # Example shutdown logic (commented out):
    # global mt5
    # if hasattr(mt5, 'shutdown'):
    #   mt5.shutdown()
    #   sd_logger.info("MT5 connection shut down (Placeholder).")
    pass

def get_live_data(symbol: str = "XAUUSD", timeframe_mt5_const = None, count: int = 100) -> pd.DataFrame | None: # timeframe changed to mt5_const
    """
    Placeholder function to get live market data from MT5.
    (Currently returns None).

    Args:
        symbol (str): Trading symbol.
        timeframe_mt5_const: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1).
        count (int): Number of bars to retrieve.
    """
    live_data_logger = logging.getLogger(f"{__name__}.MT5Connector.get_live_data") # Specific logger
    live_data_logger.debug(f"Attempting to get live data for {symbol}, timeframe_const={timeframe_mt5_const}, count={count} (Placeholder)...")
    # Example data fetching logic (commented out):
    # global mt5 # Ensure mt5 is accessible
    # global pd # Ensure pandas is accessible
    # if timeframe_mt5_const is None:
    #     live_data_logger.warning("MT5 timeframe constant not provided, defaulting to M1 (Placeholder).")
    #     timeframe_mt5_const = mt5.TIMEFRAME_M1 # Default if not provided
    #
    # rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5_const, 0, count)
    # if rates is None:
    #     live_data_logger.error(f"Failed to get rates for {symbol}, error code = {mt5.last_error()} (Placeholder)")
    #     return None
    # elif len(rates) == 0:
    #     live_data_logger.warning(f"No rates returned for {symbol} (Count: {count}) (Placeholder)")
    #     return pd.DataFrame() # Return empty DataFrame
    # else:
    #     df = pd.DataFrame(rates)
    #     df['time'] = pd.to_datetime(df['time'], unit='s')
    #     df.set_index('time', inplace=True)
    #     # Standardize column names to match historical data if needed
    #     df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    #     live_data_logger.info(f"Successfully retrieved {len(df)} rates for {symbol} (Placeholder).")
    #     return df
    live_data_logger.warning("   MT5 get_live_data logic is currently commented out (Placeholder). Returning None.")
    return None

def execute_mt5_order(
    action_type_mt5_const, # e.g., mt5.ORDER_TYPE_BUY
    symbol: str = "XAUUSD",
    lot_size: float = 0.01,
    price: float | None = None, # For market orders, MT5 uses current price
    sl: float | None = None,
    tp: float | None = None,
    deviation: int = 10,
    magic: int = 12345,
    comment: str = "GoldAI_Enterprise"
) -> dict | None:
    """
    Placeholder function to execute a market order on MT5.
    (Currently does nothing but log).
    """
    order_logger = logging.getLogger(f"{__name__}.MT5Connector.execute_mt5_order") # Specific logger
    order_logger.info(f"Attempting to execute MT5 order (Placeholder): ActionConst={action_type_mt5_const}, Symbol={symbol}, Lot={lot_size}, SL={sl}, TP={tp}, Comment={comment}")
    # Example order execution logic (commented out):
    # global mt5 # Ensure mt5 is accessible
    # if not hasattr(mt5, 'TRADE_ACTION_DEAL'): # Basic check if mt5 object is valid
    #     order_logger.error("MT5 object not properly initialized or TRADE_ACTION_DEAL not found.")
    #     return None
    #
    # current_price_to_use = price
    # if current_price_to_use is None: # For market orders
    #     tick = mt5.symbol_info_tick(symbol)
    #     if tick is None:
    #         order_logger.error(f"Failed to get tick for {symbol} to determine market price.")
    #         return None
    #     current_price_to_use = tick.ask if action_type_mt5_const == mt5.ORDER_TYPE_BUY else tick.bid
    #
    # request = {
    #     "action": mt5.TRADE_ACTION_DEAL,
    #     "symbol": symbol,
    #     "volume": lot_size,
    #     "type": action_type_mt5_const,
    #     "price": current_price_to_use,
    #     "sl": sl if sl is not None else 0.0, # MT5 expects 0.0 if no SL/TP
    #     "tp": tp if tp is not None else 0.0,
    #     "deviation": deviation,
    #     "magic": magic,
    #     "comment": comment,
    #     "type_time": mt5.ORDER_TIME_GTC, # Good till cancelled
    #     "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel (check broker compatibility)
    # }
    # result = mt5.order_send(request)
    # if result is None:
    #     order_logger.error(f"MT5 order_send failed for {symbol}. Error code: {mt5.last_error()} (Placeholder)")
    #     return None
    # elif result.retcode != mt5.TRADE_RETCODE_DONE:
    #     order_logger.error(f"MT5 order failed: retcode={result.retcode}, comment='{result.comment}' (Placeholder)")
    #     order_logger.debug(f"Failed Order Request Details: {request}")
    #     return result._asdict() if hasattr(result, '_asdict') else vars(result) # Convert to dict
    # else:
    #     order_logger.info(f"MT5 Order executed successfully: Deal={result.deal}, Order={result.order} (Placeholder)")
    #     return result._asdict() if hasattr(result, '_asdict') else vars(result)
    order_logger.warning("   MT5 order execution logic is currently commented out (Placeholder). Returning None.")
    return None

# --- Main Live Trading Loop (Conceptual Placeholder) ---
def run_live_trading_loop(config: 'StrategyConfig'): # type: ignore
    """
    Conceptual placeholder for the main live trading loop.
    Uses StrategyConfig for parameters.
    """
    live_loop_logger = logging.getLogger(f"{__name__}.MT5Connector.run_live_trading_loop") # Specific logger
    live_loop_logger.info("Starting Live Trading Loop (Conceptual Placeholder)...")

    if not initialize_mt5(config): # Pass config to initialize
        live_loop_logger.critical("Cannot start live trading loop: MT5 initialization failed.")
        return

    # Initialize RiskManager and TradeManager for live trading
    # risk_manager_live = RiskManager(config) # Example
    # trade_manager_live = TradeManager(config, risk_manager_live) # Example

    try:
        while True: # This loop would need a proper exit condition in a real system
            live_loop_logger.info("Live Loop Iteration (Placeholder)...")
            # 1. Get Live Data
            # live_data_df = get_live_data(symbol="XAUUSD", timeframe_mt5_const=mt5.TIMEFRAME_M1, count=500) # Example
            # if live_data_df is None or live_data_df.empty:
            #     live_loop_logger.warning("Could not get live data, sleeping...")
            #     time.sleep(config.live_data_poll_interval_seconds) # Example: poll interval from config
            #     continue

            # 2. Calculate Features & Signals (using live_data_df and config)
            # features_live = engineer_m1_features(live_data_df, config, lag_features_config=None)
            # signals_live = calculate_m1_entry_signals(features_live, fold_specific_config={}, strategy_config=config) # fold_specific might be empty or from a live config

            # 3. Make Trading Decision (based on signals_live.iloc[-1])
            # decision = "HOLD"
            # if signals_live.iloc[-1]['Entry_Long'] == 1: decision = "BUY"
            # elif signals_live.iloc[-1]['Entry_Short'] == 1: decision = "SELL"

            # 4. Apply ML Filter (if enabled in config, load live models)
            # if config.use_meta_classifier and decision != "HOLD":
            #     # Load models (this should be done once at startup, not in loop ideally)
            #     # available_live_models = load_live_models(config)
            #     # selected_model_key, confidence = select_model_for_trade(context_live, available_live_models)
            #     # ... ML prediction ...
            #     pass

            # 5. Execute Order (if decision is BUY/SELL and passes all checks)
            # if decision == "BUY":
            #     execute_mt5_order(action_type_mt5_const=mt5.ORDER_TYPE_BUY, lot_size=config.default_live_lot_size, ...)
            # elif decision == "SELL":
            #     execute_mt5_order(action_type_mt5_const=mt5.ORDER_TYPE_SELL, ...)

            # 6. Manage Open Positions (SL/TP updates, TSL) - This is complex
            # manage_open_positions_live(config)

            live_loop_logger.info("Live loop iteration complete (Placeholder). Sleeping...")
            time.sleep(getattr(config, 'live_trading_loop_sleep_seconds', 60)) # Sleep interval from config

    except KeyboardInterrupt:
        live_loop_logger.info("Live trading loop interrupted by user.")
    except Exception as e_live_loop: # Renamed
        live_loop_logger.critical(f"Critical error in live trading loop: {e_live_loop}", exc_info=True)
    finally:
        shutdown_mt5()
        live_loop_logger.info("Live Trading Loop Finished.")

mt5_logger.info("Part 12: MT5 Connector (Placeholder) Loaded.")
# === END OF PART 12/15 ===
# ==============================================================================
# === PART 13/15: Script Entry Point (v4.9.0 - Enterprise Refactor) ===
# ==============================================================================
# <<< MODIFIED: Enterprise Refactor - main() now called with config_file path. >>>
# <<< Log analysis pathing adjusted to use OUTPUT_DIR potentially set by config. >>>
# <<< MODIFIED: [Patch - IMPORT ERROR FIX - Step MainBlock] Added setup_gpu_acceleration() call here. >>>

import logging # Already imported
import os # Already imported
import sys # Already imported
import time # Already imported
import pandas as pd # Already imported (or dummy if import failed)
import traceback # Already imported
# main function is defined in Part 11
# run_log_analysis_pipeline is defined in Part 10 (as placeholder)
# StrategyConfig related globals (like OUTPUT_DIR) are set/updated within main()
# setup_gpu_acceleration is defined in Part 1

if __name__ == "__main__":
    # This logger is for the __main__ block itself.
    main_entry_logger = logging.getLogger(f"{__name__}.EntryPoint") # Using __name__ for the entry point logger
    start_time_script_entry = time.time()
    # Initial log to console, FileHandler will be set up inside main()
    main_entry_logger.info(f"(Starting) Script Gold Trading AI v{MINIMAL_SCRIPT_VERSION} (Enterprise Refactor)...")

    # [Patch - IMPORT ERROR FIX - Step 1 & 4] Call setup_gpu_acceleration here
    # This ensures GPU setup (including pynvml import) happens only when script is run directly.
    # It also happens *before* main() is called, so main() can potentially use USE_GPU_ACCELERATION
    # if it were to influence config loading or early decisions (though currently it doesn't seem to).
    # The primary benefit is avoiding this during test imports.
    try:
        if 'setup_gpu_acceleration' in globals() and callable(setup_gpu_acceleration):
            logger.info("[Patch - IMPORT ERROR FIX - Step MainBlock] Calling setup_gpu_acceleration() from __main__ block.")
            setup_gpu_acceleration()
            if 'print_gpu_utilization' in globals() and callable(print_gpu_utilization): # pragma: no cover
                print_gpu_utilization("After GPU Setup in __main__")
        else: # pragma: no cover
            main_entry_logger.warning("setup_gpu_acceleration function not found. GPU setup might be incomplete.")
    except Exception as e_gpu_setup_main_block: # pragma: no cover
        main_entry_logger.error(f"Error during setup_gpu_acceleration in __main__: {e_gpu_setup_main_block}", exc_info=True)


    # --- Determine Run Mode and Configuration File ---
    selected_run_mode_entry = 'FULL_PIPELINE'
    # selected_run_mode_entry = 'PREPARE_TRAIN_DATA'
    # selected_run_mode_entry = 'TRAIN_MODEL_ONLY'
    # selected_run_mode_entry = 'FULL_RUN'

    config_file_to_use = "config.yaml"

    main_entry_logger.info(f"(Starting) กำลังเริ่มการทำงานหลัก (main) ในโหมด: {selected_run_mode_entry} ด้วย Config: '{config_file_to_use}'...")
    final_run_suffix_from_main = None

    try:
        tuning_mode_used_main_entry = "Fixed Params (Default)" # Default, might be updated by main logic if applicable

        # Call the main execution function
        final_run_suffix_from_main = main(run_mode=selected_run_mode_entry, config_file=config_file_to_use)

        # --- Post-Run Analysis (if applicable) ---
        if selected_run_mode_entry not in ['TRAIN_MODEL_ONLY', 'PREPARE_TRAIN_DATA'] and final_run_suffix_from_main is not None: # pragma: no cover
            main_entry_logger.info("\n--- (Post-Run) Starting Log Analysis ---")
            output_dir_for_log_analysis = globals().get('OUTPUT_DIR') # OUTPUT_DIR is set within main()

            if output_dir_for_log_analysis and os.path.isdir(output_dir_for_log_analysis) and \
               final_run_suffix_from_main and final_run_suffix_from_main not in ["_skipped", "_no_data", "_train_only", "_train_skipped_empty_log"]:

                log_suffix_to_analyze_entry = final_run_suffix_from_main
                
                # Use config_main_obj (set in main) to get trade_log_filename_prefix
                log_filename_prefix_from_config = "trade_log" # Default
                if 'config_main_obj' in globals() and globals()['config_main_obj'] is not None:
                    log_filename_prefix_from_config = getattr(globals()['config_main_obj'], 'trade_log_filename_prefix', "trade_log")

                # Construct log path using the prefix from config and the suffix from the run
                # This assumes WFV saves with a pattern like {prefix}{suffix}.csv.gz
                # The actual WFV log saving uses a timestamp, so this might need adjustment
                # For now, let's assume a simpler pattern for this placeholder analysis trigger
                # A more robust way would be for main() or WFV to return the exact path of the log to analyze.
                
                # Attempt to find the most recent log file matching the pattern if timestamp was used
                # This is a simplified approach; a more robust method would be needed if multiple funds run.
                log_files_in_output = [f for f in os.listdir(output_dir_for_log_analysis) if f.startswith(f"{log_filename_prefix_from_config}{log_suffix_to_analyze_entry}") and f.endswith(".csv.gz")]
                log_path_to_analyze = None
                if log_files_in_output:
                    log_files_in_output.sort(key=lambda name: os.path.getmtime(os.path.join(output_dir_for_log_analysis, name)), reverse=True)
                    log_path_to_analyze = os.path.join(output_dir_for_log_analysis, log_files_in_output[0])
                    main_entry_logger.info(f"Found log file for analysis: {log_path_to_analyze}")
                else:
                    main_entry_logger.warning(f"Could not find a suitable trade log file with prefix '{log_filename_prefix_from_config}{log_suffix_to_analyze_entry}' in {output_dir_for_log_analysis}")


                if log_path_to_analyze and os.path.exists(log_path_to_analyze):
                    main_entry_logger.info(f"Analyzing log file: {log_path_to_analyze}")
                    try:
                        current_config_for_log_analysis = globals().get('config_main_obj')
                        if current_config_for_log_analysis is None:
                             main_entry_logger.warning("config_main_obj not found for log analysis, using default recovery threshold.")
                             current_config_for_log_analysis = StrategyConfig({}) # type: ignore

                        if 'run_log_analysis_pipeline' in globals() and callable(run_log_analysis_pipeline): # type: ignore
                            analysis_results_entry = run_log_analysis_pipeline( # type: ignore
                                log_file_path=log_path_to_analyze,
                                output_dir_log_analysis=output_dir_for_log_analysis,
                                config=current_config_for_log_analysis, # Pass the loaded config
                                suffix_log_analysis=log_suffix_to_analyze_entry
                            )
                            if analysis_results_entry: main_entry_logger.info("\n(Log Analysis Completed)")
                        else:
                            main_entry_logger.warning("Function 'run_log_analysis_pipeline' not found. Skipping log analysis.")
                    except Exception as e_log_analysis_entry:
                        main_entry_logger.error(f"Error during log analysis: {e_log_analysis_entry}", exc_info=True)
                else:
                    main_entry_logger.warning(f"\n(Skipping Log Analysis) Log file not found or path invalid: {log_path_to_analyze}")
            elif final_run_suffix_from_main is None:
                 main_entry_logger.warning(f"\n(Skipping Log Analysis) Main function did not return a valid suffix (returned None).")
            else:
                main_entry_logger.warning(f"\n(Skipping Log Analysis) Conditions not met. OUTPUT_DIR: '{globals().get('OUTPUT_DIR')}', Suffix: '{final_run_suffix_from_main}'")
        else:
            main_entry_logger.info(f"\n(Skipping Log Analysis) Run mode '{selected_run_mode_entry}' does not require log analysis, or main run suffix was invalid.")

    except SystemExit as se_main_block:
        main_entry_logger.critical(f"\n(Critical Error) สคริปต์ออกก่อนเวลา: {se_main_block}")
    except KeyboardInterrupt: # pragma: no cover
        main_entry_logger.warning("\n(Stopped) การทำงานหยุดโดยผู้ใช้ (KeyboardInterrupt).")
    except NameError as ne_main_block: # pragma: no cover
        main_entry_logger.critical(f"\n(Error) NameError in __main__ block: '{ne_main_block}'. Critical function or variable likely missing.", exc_info=True)
        traceback.print_exc()
    except Exception as e_main_general_block: # pragma: no cover
        main_entry_logger.critical("\n(Error) เกิดข้อผิดพลาดที่ไม่คาดคิดใน __main__ block:", exc_info=True)
        traceback.print_exc()
    finally:
        end_time_script_entry = time.time()
        total_duration_script_entry = end_time_script_entry - start_time_script_entry
        main_entry_logger.info(f"\n(Finished) Script Gold Trading AI v{MINIMAL_SCRIPT_VERSION} (Enterprise Refactor) เสร็จสมบูรณ์!")

        final_tuning_mode_log_entry = "Fixed Params (Default)"
        # This global might not be set if main() doesn't run or set it.
        # It's more of an illustrative example of how one might pass info out of main.
        # A better way would be for main() to return a dict of results.
        if 'tuning_mode_used' in globals() and globals()['tuning_mode_used'] is not None: # type: ignore # pragma: no cover
            final_tuning_mode_log_entry = str(globals()['tuning_mode_used']) # type: ignore
        
        main_entry_logger.info(f"   Tuning Mode ที่ใช้ (ถ้ามี): {final_tuning_mode_log_entry}")

        output_dir_final_log_entry = globals().get('OUTPUT_DIR', "Not Set/Available")
        log_filename_final_entry = globals().get('LOG_FILENAME', f'gold_ai_v{MINIMAL_SCRIPT_VERSION.split("_")[0]}_unknown_run.log')

        if output_dir_final_log_entry != "Not Set/Available" and os.path.exists(output_dir_final_log_entry):
            main_entry_logger.info(f"   ผลลัพธ์ถูกบันทึกไปที่: {output_dir_final_log_entry}")
            main_entry_logger.info(f"   ไฟล์ Log หลัก: {log_filename_final_entry}")
        elif output_dir_final_log_entry != "Not Set/Available": # pragma: no cover
            main_entry_logger.warning(f"   (Warning) Output Directory ที่คาดหวัง ({output_dir_final_log_entry}) ไม่พบ หรือไม่ได้ถูกสร้าง.")
        else: # pragma: no cover
            main_entry_logger.warning("   (Warning) ไม่สามารถกำหนด Output Directory path สำหรับ Final Summary.")

        main_entry_logger.info(f"   เวลาดำเนินการทั้งหมด: {total_duration_script_entry:.2f} วินาที ({total_duration_script_entry/60:.2f} นาที).")
        main_entry_logger.info("--- End of Script ---")

# === END OF PART 13/15 ===
# === START OF PART 14/15 ===
# ==============================================================================
# === PART 14: Additional Script Sections Placeholder (v4.9.0 - Enterprise Refactor) ===
# ==============================================================================
# This part is reserved for any future additions or utility functions that might
# not fit neatly into the preceding categories, or for further modularization.
# For now, it serves as a structural marker.

import logging # Already imported

# Logger for this specific part, if any code were to be added.
part14_logger = logging.getLogger(f"{__name__}.Part14_FutureAdditions")
part14_logger.debug("Part 14: Placeholder for Future Additions reached.")

# No functional code in this part for the current refactor.

# === END OF PART 14/15 ===# === START OF PART 15/15 ===
# ==============================================================================
# === PART 15: End of Script Marker (v4.9.0 - Enterprise Refactor) ===
# ==============================================================================
# This part serves as a clear marker for the end of the Gold AI script file.
# No functional code is placed here.
# It helps in verifying that all parts of the script have been processed or concatenated correctly.

import logging # Already imported

# Logger for this specific part.
part15_logger = logging.getLogger(f"{__name__}.Part15_EndOfScript")
part15_logger.info("Reached End of Part 15 (Definitive End of Script Marker). Gold AI script processing complete.")

# === END OF SCRIPT gold_ai2025.py ===
# === END OF PART 15/15 ===