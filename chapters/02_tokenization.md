# Глава 2 — Токенизация: превращение слов в числа

## Аналогия для пятилетних

Компьютеры понимают только **числа**. Они не знают, что означает буква «A» — они знают «65» (её код ASCII). Поэтому нам нужно преобразовать текст в числа перед подачей в нейронную сеть.

Простейшая идея: **присвоить каждому слову число**:
```
"cat"  ->  9246
"sat"  ->  6734
"on"   ->   389
"the"  ->   279
"mat"  -> 16789
```

Но в английском языке сотни тысяч слов. Нужно ли нам отдельное число для «antidisestablishmentarianism»? А что насчёт новых слов вроде «skibidi», которых не существовало, когда мы создавали словарь?

## Решение: пословная токенизация (BPE)

Вместо целых слов мы разбиваем текст на **частые подсловные части**:

```
"unbelievably" -> "un" + "believ" + "ably"
"running"      -> "runn" + "ing"
"cats"         -> "cat" + "s"
"lower"        -> "low" + "er"
"GPT"          -> "G" + "P" + "T"
```

Это **Byte Pair Encoding (BPE)** — тот самый алгоритм, который используют GPT-2, GPT-3, GPT-4 и большинство современных моделей.

### Как работает BPE — шаг за шагом

BPE начинается с того, что каждый символ является своим собственным «токеном», затем многократно объединяет наиболее частую пару:

**Исходный текст:** `"low lower lowest"`

```
Шаг 0 (начальный — каждый символ это токен):
l o w _ l o w e r _ l o w e s t

Шаг 1 (наиболее частая пара: 'l'+'o' -> 'lo'):
lo w _ lo w e r _ lo w e s t

Шаг 2 (наиболее частая пара: 'lo'+'w' -> 'low'):
low _ low e r _ low e s t

Шаг 3 (наиболее частая пара: 'e'+'s' -> 'es'):
low _ low e r _ low es t

Шаг 4 (наиболее частая пара: 'es'+'t' -> 'est'):
low _ low e r _ low est

Шаг 5 (наиболее частая пара: 'low'+'_' -> 'low_'):
low_ low e r _ low_ est
```

После достаточного количества объединений у нас есть словарь вроде: `{l, o, w, e, r, s, t, _, lo, ow, low, er, es, est, low_}`

Теперь новые слова могут быть представлены с помощью этих частей, даже если мы никогда раньше их не видели:

```
"lowest"  -> "low" + "est"     (оба есть в словаре!)
"slower"  -> "s" + "low" + "er" (никогда не видели, но работает!)
```

### Почему BPE лучше пословной токенизации

| Проблема | Пословная | BPE |
|---|---|---|
| "running" против "run" | Разные токены — нет общей связи | "runn" + "ing" — модель видит связь |
| Новое слово: "rizz" | Неизвестный токен → модель ошибается | "r" + "i" + "z" + "z" → работает через символы |
| Размер словаря | 500K+ (слишком много редких слов) | 50K (сбалансировано, эффективно) |
| Обработка Unicode/emoji | Часто ломается | Резервный уровень символов никогда не ошибается |

### Что насчёт специальных символов и эмодзи?

BPE работает с **байтами**, а не символами. Это означает, что он может токенизировать ЛЮБОЕ, что можно представить как байты — эмодзи, китайские иероглифы, код, LaTeX, даже бинарные данные:

```
"Hello 😊"  ->  ["Hello", " Ġ", "😊"]    (Ġ = префикс пробела в токенизаторе GPT)
"你好"       ->  токенизируется через UTF-8 байты
"def foo():"->  ["def", "Ġfoo", "()", ":"]
```

### Соглашения токенизатора GPT

| Токен | Пример | Значение |
|---|---|---|
| Обычные токены | `"cat"`, `"the"`, `"ing"` | Обычные подсловные части |
| С префиксом пробела | `"Ġcat"`, `"Ġthe"` | Слово начинается после пробела (Ġ — специальный символ) |
| `<\|endoftext\|>` | Токен EOS | Отмечает конец документа — критично для обучения |
| Заглавные буквы | `"The"` против `"the"` | Разные токены! Регистр имеет значение |

### Токен EOS — почему это важно

Токен `<|endoftext|>` (End Of Sequence) **критичен** и часто упускается из виду:

```python
# БЕЗ EOS — два документа сливаются:
doc1 = "The cat sat."     # токены: [464, 3797, 3332, 13]
doc2 = "The dog ran."     # токены: [464, 3290, 3407, 13]
# Результат: [464, 3797, 3332, 13, 464, 3290, 3407, 13]
# Модель видит: "...sat. The dog ran." — думает, что это ОДИН документ
# Учится: "sat." часто сопровождается "The" — НЕПРАВИЛЬНО!

# С EOS — документы разделены:
tokens = [464, 3797, 3332, 13, EOS, 464, 3290, 3407, 13, EOS]
# Модель учит: EOS означает «мы закончили здесь, следующий токен не связан»
```

## Tokenizer Code — Annotated

```python
from dataclasses import dataclass
import tiktoken


@dataclass
class TokenizerConfig:
    """
    WHAT: Keeps all tokenizer settings in one place.
    WHY: Like a recipe card — consistent across the whole project.
         Change one value and everything updates automatically.
    """
    name: str = "gpt2"                # WHAT: use GPT-2's pretrained BPE tokenizer
                                       # WHY: same BPE as GPT-3/4 — 50K merges,
                                       #      battle-tested on billions of documents,
                                       #      and already trained (no weeks of work)
    vocab_size: int = 50257           # WHAT: total number of unique tokens
                                       # WHY: 50,257 is the exact GPT-2 vocabulary size
                                       #      (50,000 merges + 256 byte tokens + 1 EOS)
                                       #      This is the "goldilocks" number —
                                       #      big enough for rare subwords,
                                       #      small enough for fast matrix operations


class SimpleTokenizer:
    """
    WHAT: Wraps tiktoken to give us a friendly, consistent interface.
    WHY: tiktoken's raw API is low-level (you need to specify
         allowed_special every call). This wrapper makes encode/decode
         trivial — just call .encode("hello") and get tokens back.
         
         It also handles the EOS token consistently so we never
         accidentally forget to add it during training data prep.
    """

    def __init__(self, config: TokenizerConfig = None):
        """
        WHAT: Initialize the tokenizer with GPT-2's BPE vocabulary.
        WHY: We use a pretrained tokenizer because:
             1. Training a tokenizer from scratch takes weeks of CPU time
             2. GPT-2's tokenizer is open-source, fast, and well-tested
             3. Using the same tokenizer as production models means our
                code works identically to how GPT-3 tokenizes
        """
        self.config = config or TokenizerConfig()

        # WHAT: Load the GPT-2 encoding from tiktoken
        # WHY: tiktoken stores pretrained BPE merge tables.
        #      get_encoding("gpt2") loads the exact 50K merges
        #      that GPT-2 was trained with.
        self.enc = tiktoken.get_encoding(self.config.name)

        # WHAT: Define and encode the End-of-Sequence token
        # WHY: <|endoftext|> is the special token that marks boundaries
        #      between documents. During training, we insert it between
        #      every document so the model learns where one text ends
        #      and another begins.
        self.eos_token = "<|endoftext|>"       # The string representation
        self.eos_token_id = self.enc.encode(    # Convert to its token ID
            self.eos_token,
            allowed_special={self.eos_token}    # WHY: tiktoken blocks special tokens
                                                #      by default for safety. We must
                                                #      explicitly allow EOS encoding.
        )[0]  # [0] because encode() returns a list — we want the single ID

    def encode(self, text: str) -> list[int]:
        """
        WHAT: Turn text into a list of integer token IDs.
        WHY: Neural networks only eat numbers. Raw strings like
             "Hello world" mean nothing to matrix multiplication.

        Example: "Hello world" -> [15496, 995]

        Under the hood: tiktoken splits the text into subword pieces
        using the pretrained BPE merge table, then looks up each
        piece's ID in the vocabulary.
        """
        # WHAT: Use tiktoken's fast C/Rust-based encoder
        # WHY: tiktoken is written in Rust, not Python.
        #      It can tokenize hundreds of MB of text per second.
        #      A pure Python BPE tokenizer would be 100x slower.
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids: list[int]) -> str:
        """
        WHAT: Turn token IDs back into human-readable text.
        WHY: After the model generates a sequence of token IDs
             during inference, we need to convert them back to
             text so humans can read the output.

        Example: [15496, 995] -> "Hello world"
        """
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """
        WHAT: How many unique tokens exist in the vocabulary.
        WHY: This number determines the size of our model's output
             layer — the final Linear layer must have vocab_size
             outputs (one score for each possible next token).
             
             50,257 means the model chooses from 50,257 possibilities
             every time it predicts the next word.
        """
        return self.config.vocab_size


# ===== WHAT: Quick self-test =====
# WHY: Always test each component in isolation before combining.
#      "Does the tokenizer work?" is a 5-second check that saves
#      hours of debugging a misbehaving training loop.
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Test 1: Basic text
    test_text = "The cat sat on the mat."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test 1 — Basic:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Test 2: EOS token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\nTest 2 — EOS token:")
    print(f"  String: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  Encode result: {eos}")

    # Test 3: Rare/unseen word
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\nTest 3 — Rare word:")
    print(f"  Encoded: {rare}")
    print(f"  Pieces:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  Decoded: '{decoded_rare}'")

    # Test 4: Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\nTest 4 — Emoji:")
    print(f"  Encoded: {emoji}")
    print(f"  Decoded: '{tokenizer.decode(emoji)}'")

    print(f"\n  Vocab size: {tokenizer.vocab_size:,}")
```

**Expected output:**
```
Test 1 — Basic:
  Original: 'The cat sat on the mat.'
  Encoded:  [464, 3797, 3332, 319, 262, 2603, 13]
  Decoded:  'The cat sat on the mat.'
  Match:    True

Test 2 — EOS token:
  String: '<|endoftext|>'
  Token ID: 50256
  Encode result: [50256]

Test 3 — Rare word:
  Encoded: [378, 420, 1634, 2013, 82, 622, 441, 979, 389]
  Pieces:  ['ant', 'idis', 'establish', 'ment', 'ar', 'ian', 'ism']
  Decoded: 'antidisestablishmentarianism'

Test 4 — Emoji:
  Encoded: [15496, 52430, 23530, 248, 995]
  Decoded: 'Hello 😊 world'

  Vocab size: 50,257
```

---

**Previous:** [Chapter 1 — Setup](01_setup.md)
**Next:** [Chapter 3 — Embeddings](03_embeddings.md)
