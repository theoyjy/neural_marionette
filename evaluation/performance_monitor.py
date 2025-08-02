#!/usr/bin/env python3
"""
系统性能监控工具
监控CPU、内存、GPU使用情况和执行时间
"""

import time
import threading
import json
from datetime import datetime
from pathlib import Path


class PerformanceMonitor:
    """系统性能监控器"""
    
    def __init__(self, monitor_interval=1.0):
        """
        初始化性能监控器
        Args:
            monitor_interval: 监控间隔（秒）
        """
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        
        # 性能数据存储
        self.performance_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'timestamps': []
        }
        
        # 检查依赖
        self.psutil_available = self._check_psutil()
        self.gpu_available = self._check_gpu_monitoring()
    
    def _check_psutil(self):
        """检查psutil是否可用"""
        try:
            import psutil
            return True
        except ImportError:
            print("Warning: psutil not available, CPU/Memory monitoring disabled")
            return False
    
    def _check_gpu_monitoring(self):
        """检查GPU监控是否可用"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            return True
        except Exception:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                       '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            except Exception:
                print("Warning: GPU monitoring not available")
                return False
    
    def _get_cpu_memory_stats(self):
        """获取CPU和内存统计"""
        if not self.psutil_available:
            return 0, 0
        
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024 ** 3)
            return cpu_percent, memory_used_gb
        except Exception as e:
            print(f"Error getting CPU/Memory stats: {e}")
            return 0, 0
    
    def _get_gpu_stats(self):
        """获取GPU统计"""
        if not self.gpu_available:
            return 0, 0
        
        try:
            # 首先尝试nvidia-ml-py3
            import nvidia_ml_py3 as nvml
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU使用率
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent = utilization.gpu
            
            # GPU内存
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used_gb = memory_info.used / (1024 ** 3)
            
            return gpu_percent, gpu_memory_used_gb
            
        except Exception:
            try:
                # 备选方案：使用nvidia-smi
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                                       '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        parts = lines[0].split(', ')
                        gpu_percent = float(parts[0])
                        gpu_memory_mb = float(parts[1])
                        gpu_memory_gb = gpu_memory_mb / 1024
                        return gpu_percent, gpu_memory_gb
            except Exception as e:
                print(f"Error getting GPU stats: {e}")
        
        return 0, 0
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            timestamp = time.time()
            cpu_percent, memory_gb = self._get_cpu_memory_stats()
            gpu_percent, gpu_memory_gb = self._get_gpu_stats()
            
            self.performance_data['timestamps'].append(timestamp)
            self.performance_data['cpu_usage'].append(cpu_percent)
            self.performance_data['memory_usage'].append(memory_gb)
            self.performance_data['gpu_usage'].append(gpu_percent)
            self.performance_data['gpu_memory'].append(gpu_memory_gb)
            
            time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"开始性能监控...")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        print(f"停止性能监控，总时长: {self.get_total_time():.2f}秒")
    
    def get_total_time(self):
        """获取总执行时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0
    
    def get_summary_stats(self):
        """获取汇总统计"""
        if not self.performance_data['timestamps']:
            return {}
        
        def safe_stats(data):
            if not data:
                return {'avg': 0, 'max': 0, 'min': 0}
            return {
                'avg': sum(data) / len(data),
                'max': max(data),
                'min': min(data)
            }
        
        summary = {
            'total_time_seconds': self.get_total_time(),
            'monitoring_duration': self.performance_data['timestamps'][-1] - self.performance_data['timestamps'][0] if len(self.performance_data['timestamps']) > 1 else 0,
            'sample_count': len(self.performance_data['timestamps']),
            'cpu_usage_percent': safe_stats(self.performance_data['cpu_usage']),
            'memory_usage_gb': safe_stats(self.performance_data['memory_usage']),
            'gpu_usage_percent': safe_stats(self.performance_data['gpu_usage']),
            'gpu_memory_gb': safe_stats(self.performance_data['gpu_memory']),
            'timestamp': datetime.now().isoformat(),
            'psutil_available': self.psutil_available,
            'gpu_available': self.gpu_available
        }
        
        return summary
    
    def save_performance_data(self, output_file):
        """保存性能数据到文件"""
        summary = self.get_summary_stats()
        
        # 添加详细数据
        detailed_data = {
            'summary': summary,
            'detailed_data': self.performance_data
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"性能数据已保存到: {output_path}")
        return summary
    
    def print_summary(self):
        """打印性能汇总"""
        summary = self.get_summary_stats()
        
        print(f"\n{'='*50}")
        print(f"性能监控汇总")
        print(f"{'='*50}")
        print(f"总执行时间: {summary.get('total_time_seconds', 0):.2f}秒")
        print(f"监控样本数: {summary.get('sample_count', 0)}")
        
        if summary.get('psutil_available', False):
            cpu_stats = summary.get('cpu_usage_percent', {})
            mem_stats = summary.get('memory_usage_gb', {})
            print(f"CPU使用率: 平均 {cpu_stats.get('avg', 0):.1f}%, 最大 {cpu_stats.get('max', 0):.1f}%")
            print(f"内存使用: 平均 {mem_stats.get('avg', 0):.2f}GB, 最大 {mem_stats.get('max', 0):.2f}GB")
        
        if summary.get('gpu_available', False):
            gpu_stats = summary.get('gpu_usage_percent', {})
            gpu_mem_stats = summary.get('gpu_memory_gb', {})
            print(f"GPU使用率: 平均 {gpu_stats.get('avg', 0):.1f}%, 最大 {gpu_stats.get('max', 0):.1f}%")
            print(f"GPU内存: 平均 {gpu_mem_stats.get('avg', 0):.2f}GB, 最大 {gpu_mem_stats.get('max', 0):.2f}GB")
        
        print(f"{'='*50}")


def test_monitor():
    """测试监控器"""
    monitor = PerformanceMonitor(monitor_interval=0.5)
    
    print("开始测试性能监控...")
    monitor.start_monitoring()
    
    # 模拟一些工作
    time.sleep(3)
    
    monitor.stop_monitoring()
    monitor.print_summary()
    
    # 保存数据
    monitor.save_performance_data("test_performance.json")


if __name__ == "__main__":
    test_monitor()