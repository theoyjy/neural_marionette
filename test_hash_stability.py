#!/usr/bin/env python3
"""
测试哈希算法稳定性
"""

import hashlib
from pathlib import Path

def test_hash_stability():
    """测试哈希算法稳定性"""
    print("🧪 测试哈希算法稳定性")
    print("=" * 50)
    
    # 测试文件夹路径
    test_paths = [
        "D:/Code/VVEditor/Rafa_Approves_hd_4k",
        "D:/Code/VVEditor/Another_Folder",
        "C:/Users/Test/MyData"
    ]
    
    print("📋 测试结果:")
    for folder_path in test_paths:
        folder_path_obj = Path(folder_path)
        folder_str = str(folder_path_obj.absolute())
        
        # 使用MD5哈希
        folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]
        
        # 多次计算确保稳定性
        hashes = []
        for i in range(5):
            hash_val = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]
            hashes.append(hash_val)
        
        # 检查是否所有哈希值都相同
        is_stable = len(set(hashes)) == 1
        
        print(f"\n📁 文件夹: {folder_path}")
        print(f"  - 绝对路径: {folder_path_obj.absolute()}")
        print(f"  - 哈希值: {folder_hash}")
        print(f"  - 稳定性: {'✅ 稳定' if is_stable else '❌ 不稳定'}")
        
        if not is_stable:
            print(f"  - 多次哈希结果: {hashes}")
    
    print(f"\n✅ 哈希稳定性测试完成")

def test_old_vs_new_hash():
    """对比新旧哈希算法"""
    print("\n🔄 对比新旧哈希算法")
    print("=" * 50)
    
    test_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    folder_path_obj = Path(test_path)
    folder_str = str(folder_path_obj.absolute())
    
    # 旧方法（不稳定）
    old_hash = str(hash(folder_str))[-8:]
    
    # 新方法（稳定）
    new_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]
    
    print(f"📁 测试路径: {test_path}")
    print(f"  - 旧哈希方法: {old_hash}")
    print(f"  - 新哈希方法: {new_hash}")
    print(f"  - 是否相同: {'✅' if old_hash == new_hash else '❌'}")
    
    # 多次测试旧方法
    old_hashes = []
    for i in range(3):
        old_hash_test = str(hash(folder_str))[-8:]
        old_hashes.append(old_hash_test)
    
    print(f"  - 旧方法多次结果: {old_hashes}")
    print(f"  - 旧方法稳定性: {'❌ 不稳定' if len(set(old_hashes)) > 1 else '✅ 稳定'}")

if __name__ == "__main__":
    test_hash_stability()
    test_old_vs_new_hash()
    
    print("\n📝 总结:")
    print("  - 新方法使用MD5哈希，确保相同路径总是生成相同的哈希值")
    print("  - 旧方法使用Python内置hash()，每次运行可能产生不同结果")
    print("  - 修复后，相同文件夹的多次运行将使用相同的输出目录") 