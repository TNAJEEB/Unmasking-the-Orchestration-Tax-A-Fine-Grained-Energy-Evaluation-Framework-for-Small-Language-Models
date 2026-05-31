# telemetry.py
import pynvml
import time
import threading
import os

class FullSystemMonitor:
    def __init__(self, gpu_indices=[0]):
        try:
            pynvml.nvmlInit()
        except Exception as e:
            print(f"Warning: NVML Init failed. GPU telemetry will not work. Error: {e}")
        
        self.gpu_indices = gpu_indices
        self.handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}
        self.monitoring = False
        self.energy_data = {f"GPU_{i}": 0.0 for i in gpu_indices}
        
        self.cpu_start_energy = 0.0
        self.cpu_end_energy = 0.0
        self.ram_start_energy = 0.0
        self.ram_end_energy = 0.0
        
        self.thread = None

        # Hardcoded standard path for Intel CPU Package RAPL
        self.cpu_rapl_path = '/sys/class/powercap/intel-rapl:0/energy_uj'
        # Dynamically find the RAM path (it shifts depending on motherboard architecture)
        self.dram_rapl_path = self._find_dram_path()

    def _find_dram_path(self):
        """Hunts for the specific folder containing DRAM energy data."""
        base_dir = "/sys/class/powercap/intel-rapl/intel-rapl:0/"
        if not os.path.exists(base_dir):
            return None
        try:
            for folder in os.listdir(base_dir):
                if folder.startswith("intel-rapl:0:"):
                    name_path = os.path.join(base_dir, folder, "name")
                    if os.path.exists(name_path):
                        with open(name_path, 'r') as f:
                            name = f.read().strip()
                            if "dram" in name.lower():
                                return os.path.join(base_dir, folder, "energy_uj")
        except PermissionError:
            print("\nCRITICAL PERMISSION ERROR: Cannot access RAPL files.")
            print("You MUST run this script with 'sudo' to read CPU/RAM energy.\n")
        return None

    def _get_rapl_energy_joules(self, path):
        """Reads the RAPL file and converts microjoules to Joules."""
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    # RAPL files return microjoules (uJ). Divide by 1,000,000 for Joules.
                    return int(f.read()) / 1_000_000.0
            except PermissionError:
                return 0.0
            except Exception:
                return 0.0
        return 0.0

    def _monitor_loop(self):
        """Background thread that continuously polls GPU energy via NVML."""
        last_time = time.time()
        while self.monitoring:
            current_time = time.time()
            elapsed_time = current_time - last_time
            last_time = current_time
            
            for i, handle in self.handles.items():
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle) 
                self.energy_data[f"GPU_{i}"] += (power_mw * elapsed_time) / 1000.0
            
            # Sleep 10ms to prevent the telemetry thread from eating CPU cycles
            time.sleep(0.01)

    def start(self):
        """Takes an initial snapshot of CPU/RAM and launches the GPU polling thread."""
        for key in self.energy_data.keys():
            self.energy_data[key] = 0.0
        
        self.cpu_start_energy = self._get_rapl_energy_joules(self.cpu_rapl_path)
        self.ram_start_energy = self._get_rapl_energy_joules(self.dram_rapl_path)
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        """Stops the GPU thread, takes a final snapshot of CPU/RAM, and calculates the delta."""
        self.monitoring = False
        if self.thread is not None:
            self.thread.join()
        
        self.cpu_end_energy = self._get_rapl_energy_joules(self.cpu_rapl_path)
        self.ram_end_energy = self._get_rapl_energy_joules(self.dram_rapl_path)
        
        results = {}
        # Sum all GPUs (in your case, just the RTX 5060 Ti)
        results["Total_GPU_Joules"] = sum(self.energy_data.values())
        
        # CPU and RAM energy is the exact delta between stop() and start() snapshots
        results["CPU_Joules"] = max(0.0, self.cpu_end_energy - self.cpu_start_energy)
        results["RAM_Joules"] = max(0.0, self.ram_end_energy - self.ram_start_energy)
        
        return results
