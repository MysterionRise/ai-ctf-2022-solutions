import json

data = json.load(open("token_to_id_bck.json", "r"))

for i in range(0, 35):
    print(i)
    start = i
    for key in data:
        data[key] = start
        start = (start + 1) % 35
    with open(f'token_to_id_{i}.json', 'w') as f:
        json.dump(data, f)
