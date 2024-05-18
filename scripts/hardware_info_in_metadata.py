import gradio as gr
import torch, cpuinfo, re, psutil, time
from modules.processing import StableDiffusionProcessing, Processed
from modules import errors, scripts, script_callbacks


def makeHardwareInfo():
    FORBIDDEN_WORDS = ('nvidia', 'geforce', '(r)', '(tm)', '(c)', 'cpu', 'gpu', '@',
                        'amd', 'vega', 'ryzen', 'radeon', 'intel', 'core', 'arc')

    def replace(string, old, new):
        compiled = re.compile(old, re.IGNORECASE)
        res = compiled.sub(new, string)
        return str(res)

    gpuProp = torch.cuda.get_device_properties(torch.cuda.device(0))
    gpu = gpuProp.name
    vram = f'{gpuProp.total_memory/1024/1024/1024:.0f}GB'
    cpu = cpuinfo.get_cpu_info()['brand_raw']
    ram = f'{psutil.virtual_memory().total/1024/1024/1024:.0f}GB RAM'

    hardwareInfo = f'{gpu} {vram}, {cpu}, {ram}'

    for word in FORBIDDEN_WORDS:
        hardwareInfo = replace(hardwareInfo, re.escape(word), '')
    hardwareInfo = replace(hardwareInfo, r'\s+', ' ').strip()

    return hardwareInfo


try:
    HARDWARE_INFO = makeHardwareInfo()
    OLD_GPU = HARDWARE_INFO.split(',')[0]
except Exception:
    errors.report("Can't make hardware info for metadata", exc_info=True)
    HARDWARE_INFO = OLD_GPU = "unknown"


replacedGpusTimes = 0

def replaceUsersGPU(infotext: str, params: dict):
    try:
        if newHardware := params.get('Hardware Info'):
            newGpu = newHardware.split(',')[0]
            if OLD_GPU != newGpu and "unknown" not in (newGpu, OLD_GPU):
                global replacedGpusTimes
                replacedGpusTimes += 1
                if replacedGpusTimes <= 2:
                    gr.Info(f'Your graphics card {OLD_GPU} has been replaced with {newGpu}')
    except Exception:
        pass


script_callbacks.on_infotext_pasted(replaceUsersGPU)


class Script(scripts.Script):
    def __init__(self):
        self.start = None
        self.generated = 0

    def title(self):
        return "Hardware Info in metadata"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def before_process_batch(self, *args, **kwargs):
        self.start = time.perf_counter()

    def getElapsedTime(self, p: StableDiffusionProcessing):
        elapsed = time.perf_counter() - self.start
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. " + elapsed_text
        if self.generated % p.batch_size == 0:
            self.start = time.perf_counter()
        return elapsed_text

    def postprocess_image(self, p: StableDiffusionProcessing, processed: Processed):
        self.generated += 1
        p.extra_generation_params["Hardware Info"] = HARDWARE_INFO
        p.extra_generation_params["Time taken"] = self.getElapsedTime(p)
