import yaml

with open('monai_use/input.yaml', 'r') as file:
    data = yaml.safe_load(file)

# 打印读取的数据
print(data)