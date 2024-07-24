
# -- modules --
from . import scatter_labels as scatter_labels_f
from . import scatter_tensor as scatter_tensor_f
from . import gather_tensor as gather_tensor_f

# -- functional api --
scatter_tensor = scatter_tensor_f.apply
gather_tensor = gather_tensor_f.run
scatter_labels = scatter_labels_f.run
scatter_topk = scatter_tensor_f.run_topk

