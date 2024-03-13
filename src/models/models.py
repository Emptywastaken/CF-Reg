from carla import MLModel

class MyOwnModel(MLModel):
    def __init__(self, data):
        super().__init__(data)

        self._mymodel = None

    @property
    def feature_input_order(self):
        return [...]

    @property
    def backend(self):
        return "pytorch"

    @property
    def raw_model(self):
        return self._mymodel

     # The predict function outputs
     # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

     # The predict_proba method outputs
     # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)