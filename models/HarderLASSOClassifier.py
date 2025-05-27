from .base import HarderLASSOModel
from tasks import ClassifierTaskMixin
from QUT import ClassificationQUT

class HarderLASSOClassifier(
    HarderLASSOModel,
    ClassifierTaskMixin
):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (20,),
        lambda_qut: float | None = None,
        nu: float = 0.1
    ):
        HarderLASSOModel.__init__(
            self,
            hidden_dims = hidden_dims,
            output_dim = None,
            bias = True,
            lambda_qut = lambda_qut,
            nu = nu
        )

        ClassifierTaskMixin.__init__(self)

        self.QUT = ClassificationQUT(
            lambda_qut = lambda_qut,
        )
