import pandas as pd
from models import build_cnn, compile_model

model = compile_model(build_cnn())
model.summary()

rows = [{"layer": l.name,
         "type": type(l).__name__,
         "output_shape": str(tuple(l.output.shape)),
         "params": l.count_params()} for l in model.layers]
table = pd.DataFrame(rows)
table.to_csv("reports/cnn_layers.csv", index=False)
print(table.to_string(index=False))
print("Total params:", model.count_params())