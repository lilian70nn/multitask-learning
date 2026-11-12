from run import GastroNet

# Classification
cls = GastroNet(task="classification")
result = cls("test.jpg")

# Segmentation
seg = GastroNet(task="segmentation")
mask = seg("test.jpg")

# Captioning
cap = GastroNet(task="captioning", baseline=2)
caption = cap("test.jpg")