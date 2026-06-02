```python
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.lstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
```
* Пояснения
```python
self.emb = nn.Embedding(vocab_size, emb_dim)
```
Создаёт таблицу эмбеддингов размером vocab_size × emb_dim (например 39274 × 128). Это просто большая матрица — каждая строка это вектор для одного слова. Когда приходит индекс токена 42, она возвращает строку номер 42.

```python
self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
```
Создаёт LSTM с входом размером 128 (выход эмбеддинга), скрытым состоянием 256, двумя слоями. batch_first=True первое измерение тензора — это batch, а не seq_len 

```python
self.fc = nn.Linear(hidden_dim, vocab_size)
```
Создаёт матрицу 256 × 39274. На каждой позиции последовательности берёт вектор 256 из LSTM и превращает его в 39274 логитов — по одному на каждое слово словаря.
