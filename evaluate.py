import torch

def compute_mape(var, var_hat):
    return torch.sum(torch.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return torch.sqrt(torch.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mae(var, var_hat):
    return torch.mean(torch.abs(var - var_hat))

def rescale(x, max_value, min_value):
    return (max_value - min_value) * x + min_value

def evaluate(flow_origin, flow_imputation, flow_missing_mask, print_value=True, data_norm=True):
    """评估补全精度

    Args:
        flow_origin (_type_): 原始数据
        flow_imputation (_type_): 补全数据
        flow_missing_mask (_type_): 缺失的位置
        print_value (bool, optional): 是否打印输出结果. Defaults to True.
        data_norm (bool, optional): flow_imputation数据是否归一化. Defaults to True.
    """
    pos_test = (flow_origin != 0) & (flow_missing_mask)
    if data_norm:
        # 把归一化的数据转为原始流量数据
        min_value = flow_origin.min()
        max_value = flow_origin.max()
        pred = rescale(flow_imputation, max_value, min_value)
    else:
        pred = flow_imputation
    rmse = compute_rmse(flow_origin[pos_test], pred[pos_test])
    mae = compute_mae(flow_origin[pos_test], pred[pos_test])
    mape = compute_mape(flow_origin[pos_test], pred[pos_test])*100
    if print_value:
        print('Imputation RMSE: {:.5}'.format(rmse))
        print('Imputation MAE: {:.5}'.format(mae))
        print('Imputation MAPE: {:.5}'.format(mape))
    return rmse, mae, mape