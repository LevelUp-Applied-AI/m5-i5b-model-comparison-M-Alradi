from model_selector import ModelSelector

selector = ModelSelector("config_base.json")
df = selector.run()

print(df.head())