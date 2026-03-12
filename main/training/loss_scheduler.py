


class LossScheduler:


    def __init__(self, base_weights: dict, stage_overrides: dict | None = None):

        self.base_weights = {

            "lambda_saml": base_weights.get("lambda_saml", 0.1),

            "lambda_style": base_weights.get("lambda_style", 0.01),

            "lambda_contrast": base_weights.get("lambda_contrast", 0.1),

            "lambda_tstm": base_weights.get("lambda_tstm", 1.0),

            "auxiliary_loss_freq": base_weights.get("auxiliary_loss_freq", 4),

        }

        self.stage_overrides = stage_overrides or {}


    def get_weights(self, stage: int) -> dict:

        weights = self.base_weights.copy()


        

        if stage in self.stage_overrides:

            weights.update(self.stage_overrides[stage])


        

        if stage < 2:

            weights["lambda_saml"] = 0.0

        if stage < 3:

            weights["lambda_tstm"] = 0.0


        return weights

