
# 量子文字生成模型穩定比較報告

## 實驗概覽
- **實驗時間**: 2025-08-22 14:36:11
- **訓練輪數**: 8 epochs
- **數據集數量**: 5
- **模型數量**: 2個
- **配置**: 穩定的小型配置（64維隱藏層，較少量子比特）

## 實驗結果

### 按模型類型比較

#### QKSAN 模型
| 數據集 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|--------|-------------|--------|------------|
| simple_sentences | 3.5068 | inf | 927.7 |
| short_stories | 5.1187 | 160.5802 | 930.3 |
| quantum_phrases | 2.8729 | 16.5880 | 927.8 |
| haiku | 3.8616 | 46.9156 | 930.2 |
| proverbs | 3.5441 | 33.7420 | 936.5 |

#### QRWKV 模型
| 數據集 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|--------|-------------|--------|------------|
| simple_sentences | 0.8632 | inf | 2.6 |
| short_stories | 0.9916 | 3.3380 | 2.6 |
| quantum_phrases | 0.7285 | 2.1683 | 2.5 |
| haiku | 0.1117 | 1.1082 | 2.6 |
| proverbs | 0.1407 | 2.1838 | 2.5 |

### 按數據集比較

#### simple_sentences
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 3.5068 | inf | 927.7 |
| qrwkv | 0.8632 | inf | 2.6 |

#### short_stories
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 5.1187 | 160.5802 | 930.3 |
| qrwkv | 0.9916 | 3.3380 | 2.6 |

#### quantum_phrases
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 2.8729 | 16.5880 | 927.8 |
| qrwkv | 0.7285 | 2.1683 | 2.5 |

#### haiku
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 3.8616 | 46.9156 | 930.2 |
| qrwkv | 0.1117 | 1.1082 | 2.6 |

#### proverbs
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 3.5441 | 33.7420 | 936.5 |
| qrwkv | 0.1407 | 2.1838 | 2.5 |

### 生成文本示例

#### QKSAN 生成示例
**simple_sentences**:
- the near near near near near near near near near near near near near near near near near near near near
- <UNK> near near near near near near near near near near near near near near near near near near near near
- <UNK> near near near near near near near near near near near near near near near near near near near near

**short_stories**:
- the whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered
- <UNK> whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered
- <UNK> whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered whispered


#### QRWKV 生成示例
**simple_sentences**:
- the
- <UNK>
- <UNK>

**short_stories**:
- the
- <UNK>
- <UNK>


## 性能總結

- **QKSAN**: 平均最佳驗證損失=3.7808, 平均困惑度=inf, 平均訓練時間=930.5秒
- **QRWKV**: 平均最佳驗證損失=0.5672, 平均困惑度=inf, 平均訓練時間=2.6秒
