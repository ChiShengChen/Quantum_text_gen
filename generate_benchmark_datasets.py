#!/usr/bin/env python3
"""
生成benchmark數據集腳本
為每個數據集生成2000個樣本並保存為CSV格式
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
from typing import List, Dict, Tuple
import json

# 導入數據生成模塊
from data.datasets import DatasetGenerator, load_dataset
from config.data_configs import SAMPLE_DATA, QUANTUM_DATASETS

def set_random_seed(seed: int = 42):
    """設置隨機種子確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)

def generate_large_dataset(dataset_name: str, n_samples: int = 2000) -> List[str]:
    """為指定數據集生成大量樣本"""
    print(f"🔄 生成 {dataset_name} 數據集 ({n_samples} 個樣本)...")
    
    if dataset_name == 'simple_sentences':
        return DatasetGenerator.generate_simple_sentences(n_samples)
    elif dataset_name == 'quantum_phrases':
        return DatasetGenerator.generate_quantum_phrases(n_samples)
    elif dataset_name == 'short_stories':
        # 擴展短故事數據集
        base_data = SAMPLE_DATA[dataset_name]
        return DatasetGenerator.expand_dataset(base_data, n_samples)
    elif dataset_name == 'haiku':
        # 擴展俳句數據集
        base_data = SAMPLE_DATA[dataset_name]
        return DatasetGenerator.expand_dataset(base_data, n_samples)
    elif dataset_name == 'proverbs':
        # 擴展諺語數據集
        base_data = SAMPLE_DATA[dataset_name]
        return DatasetGenerator.expand_dataset(base_data, n_samples)
    else:
        raise ValueError(f"不支持的數據集: {dataset_name}")

def create_train_val_test_splits(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """將數據集分割為訓練、驗證和測試集"""
    n_total = len(texts)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]
    
    return train_texts, val_texts, test_texts

def save_dataset_to_csv(texts: List[str], filename: str, dataset_info: Dict):
    """保存數據集為CSV格式"""
    df = pd.DataFrame({
        'text': texts,
        'length': [len(text.split()) for text in texts],
        'char_count': [len(text) for text in texts]
    })
    
    # 添加數據集信息
    df['dataset_name'] = dataset_info['name']
    df['dataset_description'] = dataset_info['description']
    df['source'] = dataset_info['source']
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"✅ 保存到: {filename} ({len(texts)} 個樣本)")

def generate_all_benchmark_datasets(n_samples: int = 2000, output_dir: str = "benchmark_datasets"):
    """生成所有benchmark數據集"""
    print(f"🚀 開始生成benchmark數據集")
    print(f"📊 每個數據集: {n_samples} 個樣本")
    print(f"📁 輸出目錄: {output_dir}")
    print("="*60)
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置隨機種子確保可重現性
    set_random_seed(42)
    
    # 生成時間戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 數據集統計信息
    dataset_stats = {}
    
    # 為每個數據集生成樣本
    for dataset_name in QUANTUM_DATASETS.keys():
        print(f"\n📝 處理數據集: {dataset_name}")
        
        try:
            # 生成完整數據集
            all_texts = generate_large_dataset(dataset_name, n_samples)
            
            # 分割數據集
            train_texts, val_texts, test_texts = create_train_val_test_splits(all_texts)
            
            # 獲取數據集信息
            dataset_info = QUANTUM_DATASETS[dataset_name]
            
            # 保存完整數據集
            full_filename = os.path.join(output_dir, f"{dataset_name}_full_{timestamp}.csv")
            save_dataset_to_csv(all_texts, full_filename, dataset_info.__dict__)
            
            # 保存訓練集
            train_filename = os.path.join(output_dir, f"{dataset_name}_train_{timestamp}.csv")
            save_dataset_to_csv(train_texts, train_filename, dataset_info.__dict__)
            
            # 保存驗證集
            val_filename = os.path.join(output_dir, f"{dataset_name}_val_{timestamp}.csv")
            save_dataset_to_csv(val_texts, val_filename, dataset_info.__dict__)
            
            # 保存測試集
            test_filename = os.path.join(output_dir, f"{dataset_name}_test_{timestamp}.csv")
            save_dataset_to_csv(test_texts, test_filename, dataset_info.__dict__)
            
            # 記錄統計信息
            dataset_stats[dataset_name] = {
                'total_samples': len(all_texts),
                'train_samples': len(train_texts),
                'val_samples': len(val_texts),
                'test_samples': len(test_texts),
                'avg_length': np.mean([len(text.split()) for text in all_texts]),
                'avg_char_count': np.mean([len(text) for text in all_texts]),
                'files': {
                    'full': full_filename,
                    'train': train_filename,
                    'val': val_filename,
                    'test': test_filename
                }
            }
            
            print(f"✅ {dataset_name} 完成")
            
        except Exception as e:
            print(f"❌ {dataset_name} 生成失敗: {e}")
            continue
    
    # 保存統計信息
    stats_filename = os.path.join(output_dir, f"dataset_statistics_{timestamp}.json")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
    
    # 生成README文件
    generate_readme(output_dir, timestamp, dataset_stats)
    
    print(f"\n🎉 所有數據集生成完成!")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"📊 統計信息: {stats_filename}")
    
    return dataset_stats

def generate_readme(output_dir: str, timestamp: str, dataset_stats: Dict):
    """生成README文件"""
    readme_content = f"""# Benchmark Datasets for Quantum Text Generation

## 概述
這些數據集是為量子文字生成模型benchmark而生成的，每個數據集包含2000個樣本。

## 生成信息
- **生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **樣本數量**: 每個數據集2000個樣本
- **分割比例**: 80% 訓練, 10% 驗證, 10% 測試
- **隨機種子**: 42 (確保可重現性)

## 數據集列表

"""
    
    for dataset_name, stats in dataset_stats.items():
        dataset_info = QUANTUM_DATASETS[dataset_name]
        readme_content += f"""### {dataset_name}
- **描述**: {dataset_info.description}
- **總樣本數**: {stats['total_samples']}
- **訓練集**: {stats['train_samples']} 樣本
- **驗證集**: {stats['val_samples']} 樣本  
- **測試集**: {stats['test_samples']} 樣本
- **平均長度**: {stats['avg_length']:.1f} 詞
- **平均字符數**: {stats['avg_char_count']:.1f}
- **來源**: {dataset_info.source}

"""
    
    readme_content += f"""## 文件格式

每個數據集包含以下CSV文件：
- `{dataset_name}_full_{timestamp}.csv` - 完整數據集
- `{dataset_name}_train_{timestamp}.csv` - 訓練集
- `{dataset_name}_val_{timestamp}.csv` - 驗證集
- `{dataset_name}_test_{timestamp}.csv` - 測試集

### CSV列說明
- `text`: 文本內容
- `length`: 詞數
- `char_count`: 字符數
- `dataset_name`: 數據集名稱
- `dataset_description`: 數據集描述
- `source`: 數據來源

## 使用方法

### Python (pandas)
```python
import pandas as pd

# 讀取訓練集
train_df = pd.read_csv('{dataset_name}_train_{timestamp}.csv')

# 讀取文本
texts = train_df['text'].tolist()
```

### Python (PyTorch)
```python
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df['text'].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# 使用
train_dataset = TextDataset('{dataset_name}_train_{timestamp}.csv')
```

## 數據集統計

詳細統計信息請查看 `dataset_statistics_{timestamp}.json` 文件。

## 注意事項

1. 所有數據集使用相同的隨機種子生成，確保可重現性
2. 數據集已按8:1:1的比例分割為訓練、驗證、測試集
3. 文本長度已標準化，適合量子計算實驗
4. 所有文本均為英文，編碼為UTF-8

## 引用

如果使用這些數據集，請引用：
```
Quantum Text Generation Benchmark Datasets
Generated on {datetime.now().strftime('%Y-%m-%d')}
```
"""
    
    readme_filename = os.path.join(output_dir, f"README_{timestamp}.md")
    with open(readme_filename, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📄 README文件已生成: {readme_filename}")

def main():
    """主函數"""
    print("🚀 量子文字生成Benchmark數據集生成器")
    print("="*50)
    
    # 配置參數
    n_samples = 2000
    output_dir = "benchmark_datasets"
    
    # 生成數據集
    stats = generate_all_benchmark_datasets(n_samples, output_dir)
    
    # 顯示總結
    print(f"\n📊 生成總結:")
    print(f"   數據集數量: {len(stats)}")
    print(f"   總樣本數: {sum(s['total_samples'] for s in stats.values())}")
    print(f"   輸出目錄: {output_dir}")
    
    print(f"\n🎯 數據集已準備好供其他專案benchmark使用!")

if __name__ == "__main__":
    main()
