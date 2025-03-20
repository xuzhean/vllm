import random

# 定义一些随机的词汇
nouns = ["cat", "dog", "tree", "house", "car", "book", "computer", "phone", "mountain", "river"]
verbs = ["run", "jump", "think", "write", "read", "swim", "fly", "dance", "sing", "paint"]
adjectives = ["happy", "sad", "angry", "beautiful", "ugly", "bright", "dark", "cold", "hot", "quiet"]
adverbs = ["quickly", "slowly", "happily", "sadly", "angrily", "quietly", "loudly", "carefully", "carelessly", "suddenly"]
prepositions = ["in", "on", "at", "by", "with", "about", "under", "over", "between", "through"]

def random_sentence():
    return random.choice([
        f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)} {random.choice(prepositions)} the {random.choice(nouns)}.",
        f"{random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)} {random.choice(prepositions)} the {random.choice(adjectives)} {random.choice(nouns)}.",
        f"Why does the {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)} {random.choice(prepositions)} the {random.choice(nouns)}?",
        f"Imagine a {random.choice(adjectives)} {random.choice(nouns)} that {random.choice(verbs)} {random.choice(adverbs)} {random.choice(prepositions)} the {random.choice(nouns)}."
    ])

# 生成随机 prompt 的函数
def generate_random_prompt(re: int = 1):
    prompt = ""
    for i in range(re):
        prompt = prompt + random_sentence()
        if re > 1:
            prompt = prompt + " | "
    return prompt

# 生成 100 条随机 prompt
# random_prompts = [generate_random_prompt() for _ in range(100)]

# 打印生成的 prompt
# for i, prompt in enumerate(random_prompts, 1):
#     print(f"{i}. {prompt}")