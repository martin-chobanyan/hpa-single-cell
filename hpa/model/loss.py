from torch import log
from torch.nn import Module


class MultiLabelNLL(Module):
    def forward(self, y_pred, y_label):
        """Main method

        Parameters
        ----------
        y_pred: torch.Tensor
            Sigmoid output with shape (n_batch, n_label)
        y_label: torch.Tensor
            Label output with ones indicating the presence of a class and zero indicating its
            abscence. The shape is (n_batch, n_label)

        Returns
        -------
        torch.tensor
            The loss as a singleton tensor value
        """
        return -1 * (y_label * log(y_pred)).sum()
