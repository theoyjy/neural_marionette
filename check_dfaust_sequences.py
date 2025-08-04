#!/usr/bin/env python3
"""
DFAUST数据集序列检查工具
检查HDF5文件中的可用序列，并生成all_sequences.json文件供batch_evaluation.py使用
"""

import h5py
import json
import os
from pathlib import Path


def parse_subjects_and_sequences():
    """解析subjects_and_sequences.txt文件"""
    subjects_file = Path("evaluation/data/dfaust/scripts/subjects_and_sequences.txt")
    
    if not subjects_file.exists():
        print(f"错误：找不到文件 {subjects_file}")
        return None
    
    subjects_data = {}
    current_subject = None
    
    with open(subjects_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是subject行（包含性别信息）
            if '(' in line and ')' in line:
                # 例如: "50002 (male)" 或 "50004 (female)"
                parts = line.split('(')
                subject_id = parts[0].strip()
                gender = parts[1].replace(')', '').strip()
                current_subject = subject_id
                subjects_data[subject_id] = {
                    'gender': gender,
                    'sequences': []
                }
            elif current_subject:
                # 这是一个序列名
                subjects_data[current_subject]['sequences'].append(line)
    
    return subjects_data


def check_hdf5_sequences(hdf5_path, subjects_data):
    """检查HDF5文件中实际存在的序列"""
    if not os.path.exists(hdf5_path):
        print(f"警告：HDF5文件不存在 {hdf5_path}")
        return {}
    
    available_sequences = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 获取HDF5文件中的所有键
            hdf5_keys = list(f.keys())
            
            # 检查每个subject的每个sequence
            for subject_id, info in subjects_data.items():
                subject_sequences = []
                
                for sequence_id in info['sequences']:
                    # DFAUST数据格式：subject_id + '_' + sequence_id
                    hdf5_key = f"{subject_id}_{sequence_id}"
                    
                    if hdf5_key in hdf5_keys:
                        subject_sequences.append(sequence_id)
                    else:
                        print(f"序列不存在于HDF5文件中: {hdf5_key}")
                
                if subject_sequences:
                    available_sequences[subject_id] = subject_sequences
                    print(f"Subject {subject_id} ({info['gender']}): {len(subject_sequences)} 个序列")
    
    except Exception as e:
        print(f"读取HDF5文件时出错 {hdf5_path}: {e}")
        return {}
    
    return available_sequences


def generate_all_sequences_json():
    """生成all_sequences.json文件"""
    
    # 1. 解析subjects_and_sequences.txt
    print("正在解析 subjects_and_sequences.txt...")
    subjects_data = parse_subjects_and_sequences()
    if not subjects_data:
        return False
    
    print(f"找到 {len(subjects_data)} 个subjects")
    
    # 2. 检查两个HDF5文件
    male_hdf5 = "evaluation/data/dfaust/registrations_m.hdf5"
    female_hdf5 = "evaluation/data/dfaust/registrations_f.hdf5"
    
    print(f"\n正在检查男性数据文件: {male_hdf5}")
    male_sequences = {}
    female_sequences = {}
    
    # 分离男性和女性subjects
    male_subjects = {k: v for k, v in subjects_data.items() if v['gender'] == 'male'}
    female_subjects = {k: v for k, v in subjects_data.items() if v['gender'] == 'female'}
    
    # 检查男性HDF5文件
    if male_subjects:
        male_sequences = check_hdf5_sequences(male_hdf5, male_subjects)
    
    print(f"\n正在检查女性数据文件: {female_hdf5}")
    # 检查女性HDF5文件
    if female_subjects:
        female_sequences = check_hdf5_sequences(female_hdf5, female_subjects)
    
    # 3. 生成最终的JSON格式
    all_sequences = {
        'male': male_sequences,
        'female': female_sequences
    }
    
    # 4. 保存到文件
    output_file = Path("evaluation/data/dfaust/all_sequences.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_sequences, f, indent=2, ensure_ascii=False)
    
    # 5. 打印统计信息
    total_male_sequences = sum(len(sequences) for sequences in male_sequences.values())
    total_female_sequences = sum(len(sequences) for sequences in female_sequences.values())
    
    print(f"\n序列检查完成！")
    print(f"男性subjects: {len(male_sequences)} 个，总计 {total_male_sequences} 个序列")
    print(f"女性subjects: {len(female_sequences)} 个，总计 {total_female_sequences} 个序列")
    print(f"输出文件: {output_file}")
    
    # 6. 显示详细信息
    print(f"\n详细信息:")
    print("男性序列:")
    for subject_id, sequences in male_sequences.items():
        print(f"  {subject_id}: {sequences}")
    
    print("女性序列:")
    for subject_id, sequences in female_sequences.items():
        print(f"  {subject_id}: {sequences}")
    
    return True


def main():
    """主函数"""
    print("DFAUST序列检查工具")
    print("=" * 50)
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查工作目录
    if not Path("evaluation/data/dfaust").exists():
        print("错误：找不到 evaluation/data/dfaust 目录")
        print("请确保在正确的项目根目录下运行此脚本")
        return
    
    # 检查必要文件
    scripts_dir = Path("evaluation/data/dfaust/scripts")
    if not scripts_dir.exists():
        print(f"错误：找不到 scripts 目录: {scripts_dir}")
        return
    
    subjects_file = scripts_dir / "subjects_and_sequences.txt"
    if not subjects_file.exists():
        print(f"错误：找不到 subjects_and_sequences.txt 文件: {subjects_file}")
        return
    
    # 开始处理
    success = generate_all_sequences_json()
    
    if success:
        print("\n✅ all_sequences.json 文件已成功生成")
        print("现在可以运行 batch_evaluation.py 进行批量评估")
    else:
        print("\n❌ 生成失败")


if __name__ == "__main__":
    main()