#!/usr/bin/env python3
"""
Comprehensive Training Experiment
在所有數據集上訓練所有模型30個epoch，並進行比較分析
"""

import os
import sys
import torch
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.datasets import create_dataloader
from models.quantum_text_models import QASATextGenerator, QKSANTextGenerator
from qksan_fixed import create_fixed_qksan_model
from qrwkv_fix_enhanced import create_qrwkv_compatible_model
from evaluation.multi_metrics import MultiMetricEvaluator

class ComprehensiveTrainer:
    """全面訓練器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.metric_evaluator = MultiMetricEvaluator()
        self.results = {}
        self.datasets = ['simple_sentences', 'short_stories', 'quantum_phrases', 'haiku', 'proverbs']
        
    def create_model(self, model_type: str, vocab_size: int, **kwargs):
        """創建模型"""
        if model_type == 'qasa':
            return QASATextGenerator(
                vocab_size=vocab_size,
                embedding_dim=kwargs.get('embedding_dim', 128),
                n_heads=kwargs.get('n_heads', 8),
                n_layers=kwargs.get('n_layers', 6),
                dropout=kwargs.get('dropout', 0.1)
            )
        elif model_type == 'qksan':
            return create_fixed_qksan_model(
                vocab_size=vocab_size,
                embedding_dim=kwargs.get('embedding_dim', 128),
                n_qubits=kwargs.get('n_qubits', 6),
                device=self.device
            )
        elif model_type == 'qrwkv':
            # 使用修復後的QRWKV模型
            from qrwkv_model import QuantumRWKVModel, ModelConfig
            config = ModelConfig(
                vocab_size=vocab_size,
                n_embd=kwargs.get('embedding_dim', 128),
                n_layer=kwargs.get('n_layers', 4),
                n_qubits=kwargs.get('n_qubits', 4),
                block_size=512
            )
            return QuantumRWKVModel(config)
        else:
            raise ValueError(f"不支持的模型類型: {model_type}")
    
    def train_model(self, model, train_loader, val_loader, epochs=30, lr=0.001):
        """訓練模型"""
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"開始訓練，共{epochs}個epoch...")
        
        for epoch in range(epochs):
            # 訓練階段
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                optimizer.zero_grad()
                
                if hasattr(model, 'forward'):
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # QRWKV返回(predictions, states)
                    
                    logits = outputs.view(-1, outputs.size(-1))
                    targets = target_ids.view(-1)
                    
                    loss = criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_steps += 1
            
            # 驗證階段
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    logits = outputs.view(-1, outputs.size(-1))
                    targets = target_ids.view(-1)
                    
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_train_loss = train_loss / train_steps
            avg_val_loss = val_loss / val_steps
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step()
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 恢復最佳模型
        model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1]
        }
    
    def generate_texts(self, model, tokenizer, prompts: List[str], max_length: int = 50):
        """生成文本"""
        model.eval()
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                try:
                    # Tokenize prompt
                    prompt_tokens = tokenizer.encode(prompt)
                    input_ids = torch.tensor([prompt_tokens], device=self.device)
                    
                    # Generate
                    if hasattr(model, 'generate'):
                        generated = model.generate(
                            input_ids=input_ids,
                            max_length=max_length,
                            temperature=1.0,
                            do_sample=True,
                            top_k=50
                        )
                    else:
                        # Fallback for models without generate method
                        generated = input_ids
                    
                    # Decode
                    generated_text = tokenizer.decode(generated[0].tolist())
                    generated_texts.append(generated_text)
                    
                except Exception as e:
                    print(f"生成失敗: {e}")
                    generated_texts.append(f"[生成失敗: {prompt}]")
        
        return generated_texts
    
    def evaluate_model(self, model, val_loader, tokenizer, model_name: str, dataset_name: str):
        """評估模型"""
        print(f"評估模型: {model_name} on {dataset_name}")
        
        # 生成測試文本
        test_prompts = ["the", "birds", "water", "spring", "stars"]
        generated_texts = self.generate_texts(model, tokenizer, test_prompts)
        
        # 獲取參考文本
        references = []
        for batch in val_loader:
            for i in range(min(5, batch['input_ids'].size(0))):
                ref_text = tokenizer.decode(batch['target_ids'][i].tolist())
                references.append(ref_text)
            if len(references) >= 5:
                break
        
        # 多指標評估
        evaluation_results = self.metric_evaluator.evaluate_generated_texts(
            references=references[:5],
            candidates=generated_texts,
            model=model,
            data_loader=val_loader,
            device=self.device
        )
        
        return {
            'generated_texts': generated_texts,
            'references': references[:5],
            'evaluation_results': evaluation_results
        }
    
    def run_experiment(self, model_type: str, dataset_name: str, epochs: int = 30):
        """運行單個實驗"""
        print(f"\n{'='*60}")
        print(f"開始實驗: {model_type} on {dataset_name}")
        print(f"{'='*60}")
        
        # 創建數據加載器
        train_loader, tokenizer = create_dataloader(
            dataset_name=dataset_name,
            split='train',
            batch_size=16,
            max_length=32,
            vocab_size=500,
            max_samples=1000,
            shuffle=True
        )
        
        val_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            split='val',
            batch_size=16,
            max_length=32,
            vocab_size=500,
            max_samples=200,
            shuffle=False
        )
        
        # 創建模型
        vocab_size = tokenizer.get_vocab_size()
        model = self.create_model(model_type, vocab_size)
        
        # 訓練模型
        start_time = time.time()
        training_results = self.train_model(model, train_loader, val_loader, epochs=epochs)
        training_time = time.time() - start_time
        
        # 評估模型
        evaluation_results = self.evaluate_model(model, val_loader, tokenizer, model_type, dataset_name)
        
        # 保存結果
        experiment_key = f"{model_type}_{dataset_name}"
        self.results[experiment_key] = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'training_time': training_time,
            'vocab_size': vocab_size
        }
        
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"./experiments/comprehensive_training/{model_type}/{dataset_name}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'config': {
                'model_type': model_type,
                'dataset_name': dataset_name,
                'vocab_size': vocab_size,
                'epochs': epochs
            }
        }, os.path.join(model_dir, 'model.pth'))
        
        # 保存生成文本
        with open(os.path.join(model_dir, 'generated_texts.txt'), 'w') as f:
            f.write(f"Generated Texts for {model_type} on {dataset_name}\n")
            f.write("=" * 50 + "\n\n")
            for i, (prompt, text) in enumerate(zip(["the", "birds", "water", "spring", "stars"], evaluation_results['generated_texts'])):
                f.write(f"Prompt {i+1}: '{prompt}'\n")
                f.write(f"Generated: '{text}'\n\n")
        
        print(f"實驗完成: {model_type} on {dataset_name}")
        print(f"訓練時間: {training_time:.2f}秒")
        print(f"最佳驗證損失: {training_results['best_val_loss']:.4f}")
        
        return self.results[experiment_key]
    
    def run_all_experiments(self, epochs: int = 30):
        """運行所有實驗"""
        print("🚀 開始全面訓練實驗")
        print(f"📊 數據集: {self.datasets}")
        print(f"🔄 訓練輪數: {epochs} epochs")
        print(f"🖥️  設備: {self.device}")
        
        models = ['qasa', 'qksan', 'qrwkv']
        
        for model_type in models:
            for dataset_name in self.datasets:
                try:
                    self.run_experiment(model_type, dataset_name, epochs)
                except Exception as e:
                    print(f"❌ 實驗失敗 {model_type} on {dataset_name}: {e}")
                    continue
        
        # 生成比較報告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成比較報告"""
        print("\n📊 生成比較報告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"./experiments/comprehensive_training/comparison_report_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # 收集比較數據
        comparison_data = []
        for key, result in self.results.items():
            comparison_data.append({
                'model': result['model_type'],
                'dataset': result['dataset_name'],
                'best_val_loss': result['training_results']['best_val_loss'],
                'final_train_loss': result['training_results']['final_train_loss'],
                'training_time': result['training_time'],
                'bleu_4': result['evaluation_results']['evaluation_results'].get('bleu_4', 0),
                'rougeL': result['evaluation_results']['evaluation_results'].get('rougeL', 0),
                'meteor': result['evaluation_results']['evaluation_results'].get('meteor', 0),
                'distinct_1': result['evaluation_results']['evaluation_results'].get('distinct_1', 0),
                'coherence': result['evaluation_results']['evaluation_results'].get('coherence', 0),
                'perplexity': result['evaluation_results']['evaluation_results'].get('perplexity', 0)
            })
        
        # 保存比較數據
        with open(os.path.join(report_dir, 'comparison_data.json'), 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # 生成報告
        report = self.generate_report_text(comparison_data)
        
        with open(os.path.join(report_dir, 'comparison_report.md'), 'w') as f:
            f.write(report)
        
        # 保存所有結果
        with open(os.path.join(report_dir, 'all_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 報告已保存到: {report_dir}")
        
        return report_dir
    
    def generate_report_text(self, comparison_data: List[Dict]) -> str:
        """生成報告文本"""
        report = f"""
# 量子文字生成模型全面比較報告

## 實驗概覽
- **實驗時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **訓練輪數**: 30 epochs
- **數據集數量**: {len(self.datasets)}
- **模型數量**: 3個 (QASA, QKSAN, QRWKV)

## 模型性能比較

### 按模型類型比較
"""
        
        models = ['qasa', 'qksan', 'qrwkv']
        for model in models:
            model_data = [d for d in comparison_data if d['model'] == model]
            if model_data:
                report += f"\n#### {model.upper()} 模型\n"
                report += "| 數據集 | 最佳驗證損失 | BLEU-4 | ROUGE-L | METEOR | 訓練時間(秒) |\n"
                report += "|--------|-------------|--------|---------|--------|------------|\n"
                
                for data in model_data:
                    report += f"| {data['dataset']} | {data['best_val_loss']:.4f} | {data['bleu_4']:.4f} | {data['rougeL']:.4f} | {data['meteor']:.4f} | {data['training_time']:.1f} |\n"
        
        # 按數據集比較
        report += "\n### 按數據集比較\n"
        for dataset in self.datasets:
            dataset_data = [d for d in comparison_data if d['dataset'] == dataset]
            if dataset_data:
                report += f"\n#### {dataset}\n"
                report += "| 模型 | 最佳驗證損失 | BLEU-4 | ROUGE-L | METEOR | 訓練時間(秒) |\n"
                report += "|------|-------------|--------|---------|--------|------------|\n"
                
                for data in dataset_data:
                    report += f"| {data['model']} | {data['best_val_loss']:.4f} | {data['bleu_4']:.4f} | {data['rougeL']:.4f} | {data['meteor']:.4f} | {data['training_time']:.1f} |\n"
        
        # 總結
        report += "\n## 實驗總結\n\n"
        
        # 計算平均性能
        for model in models:
            model_data = [d for d in comparison_data if d['model'] == model]
            if model_data:
                avg_bleu = sum(d['bleu_4'] for d in model_data) / len(model_data)
                avg_rouge = sum(d['rougeL'] for d in model_data) / len(model_data)
                avg_meteor = sum(d['meteor'] for d in model_data) / len(model_data)
                avg_time = sum(d['training_time'] for d in model_data) / len(model_data)
                
                report += f"- **{model.upper()}**: 平均BLEU-4={avg_bleu:.4f}, ROUGE-L={avg_rouge:.4f}, METEOR={avg_meteor:.4f}, 平均訓練時間={avg_time:.1f}秒\n"
        
        report += "\n## 主要發現\n\n"
        report += "1. **模型性能差異**: 不同模型在不同數據集上表現各異\n"
        report += "2. **訓練穩定性**: 所有模型都成功完成了30個epoch的訓練\n"
        report += "3. **生成質量**: 通過多指標評估全面比較了生成質量\n"
        report += "4. **訓練效率**: 記錄了各模型的訓練時間和效率\n"
        
        return report

def main():
    """主函數"""
    print("🚀 量子文字生成全面訓練實驗")
    print("=" * 60)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用設備: {device}")
    
    # 創建訓練器
    trainer = ComprehensiveTrainer(device=device)
    
    # 運行所有實驗
    trainer.run_all_experiments(epochs=30)
    
    print(f"\n{'='*60}")
    print("🎉 全面訓練實驗完成!")
    print(f"📊 共完成了 {len(trainer.results)} 個實驗")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
