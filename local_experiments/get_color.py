from gen_color import MunsellColor
import json

def format_prompt(pair_a: Tuple[MunsellColor, MunsellColor],
                  pair_b: Tuple[MunsellColor, MunsellColor]) -> str:
    prompt_template = (
        "You are a human with normal vision. "
        "Compare the following two pairs of colors and tell which pair looks more similar in terms of perceived color difference.\n\n"
        "Focus not on numbers, but on how different they feel to you.\n\n"
        "Pair A: {pair_a}\n"
        "Pair B: {pair_b}\n"
        "Which pair looks more similar in terms of perceived difference?"
    )
    formatted_pair_a = ", ".join(map(str, pair_a))
    formatted_pair_b = ", ".join(map(str, pair_b))
    return prompt_template.format(pair_a=formatted_pair_a, pair_b=formatted_pair_b)

def compare_colors(model_input: str) -> str:
    """
    Функция обращается к языку моделирования Qwen-2.5VL через API или библиотека HuggingFace.
    ЗАГЛУШКА - TODO - сделать не локальную версию"
    """
    print(f"Sending request: {model_input}")
    response = input("Enter model's answer here: ")  
    return response.strip()

def experiment_loop(n_experiments: int, chain_type: str='hue'):
    results = {}
    for exp_num in range(n_experiments):
        colors = MunsellColor.munsell_color_pairs(1, chain_type)[0]
        control_colors = generate_control_pairs([colors[0]], distance=2.5)[0]
        
        # Формируем задание для модели
        prompt = format_prompt(colors, control_colors)
        answer = compare_colors(prompt)
        
        # Сохраняем результаты
        results[f"Experiment_{exp_num}"] = {
            "prompt": prompt,
            "answer": answer,
            "pair_A": colors,
            "pair_B": control_colors
        }
    return results

if __name__ == "__main__":
    experiments_results = experiment_loop(n_experiments=10, chain_type='hue')
    with open('experiment_results.json', 'w') as outfile:
        json.dump(experiments_results, outfile, indent=4)
