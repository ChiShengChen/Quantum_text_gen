
# 量子文字生成模型穩定比較報告

## 實驗概覽
- **實驗時間**: 2025-08-22 14:35:46
- **訓練輪數**: 8 epochs
- **數據集數量**: 5
- **模型數量**: 2個
- **配置**: 穩定的小型配置（64維隱藏層，較少量子比特）

## 實驗結果

### 按模型類型比較

#### QKSAN 模型
| 數據集 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|--------|-------------|--------|------------|
| simple_sentences | 3.1556 | inf | 923.8 |
| short_stories | 4.4810 | 92.4960 | 925.8 |
| quantum_phrases | 3.4285 | 29.7115 | 924.1 |
| haiku | 4.0008 | 53.1198 | 926.3 |
| proverbs | 3.6323 | 36.5221 | 925.9 |

#### QRWKV 模型
| 數據集 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|--------|-------------|--------|------------|
| simple_sentences | 0.8271 | inf | 2.7 |
| short_stories | 0.9920 | 1.2146 | 2.7 |
| quantum_phrases | 0.9656 | 2.3372 | 3.0 |
| haiku | 0.1122 | 1.1101 | 3.0 |
| proverbs | 0.4241 | 1.2553 | 2.7 |

### 按數據集比較

#### simple_sentences
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 3.1556 | inf | 923.8 |
| qrwkv | 0.8271 | inf | 2.7 |

#### short_stories
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 4.4810 | 92.4960 | 925.8 |
| qrwkv | 0.9920 | 1.2146 | 2.7 |

#### quantum_phrases
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 3.4285 | 29.7115 | 924.1 |
| qrwkv | 0.9656 | 2.3372 | 3.0 |

#### haiku
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 4.0008 | 53.1198 | 926.3 |
| qrwkv | 0.1122 | 1.1101 | 3.0 |

#### proverbs
| 模型 | 最佳驗證損失 | 困惑度 | 訓練時間(秒) |
|------|-------------|--------|------------|
| qksan | 3.6323 | 36.5221 | 925.9 |
| qrwkv | 0.4241 | 1.2553 | 2.7 |

### 生成文本示例

#### QKSAN 生成示例
**simple_sentences**:
- the the the the the the the the the the the the the the the the the the the the the
- <UNK> the the the the the the the the the the the the the the the the the the the the
- <UNK> the the the the the the the the the the the the the the the the the the the the

**short_stories**:
- the
- <UNK>
- <UNK>


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

- **QKSAN**: 平均最佳驗證損失=3.7396, 平均困惑度=inf, 平均訓練時間=925.2秒
- **QRWKV**: 平均最佳驗證損失=0.6642, 平均困惑度=inf, 平均訓練時間=2.8秒
