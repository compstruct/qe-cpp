#include <torch/extension.h>

#include <iostream>
#include <cstdint>
#include <vector>
#include <map>
#include <tuple>

// torch.uint8 corresponds to torch.ByteTensor
// no parallel update support yet

// std::map<std::string, torch::Tensor> qe(torch::Tensor acc, float init, float lammy, float q) 
std::tuple<torch::Tensor, torch::Tensor> qe(torch::Tensor acc, float init, float lammy, float q) 
{
	auto mask = torch::zeros_like(acc, torch::dtype(torch::kBool));
	auto est = torch::zeros_like(acc, torch::dtype(torch::kFloat32));
	size_t acc_size = acc.numel();
	auto acc_ptr = acc.accessor<float, 1>();
	auto est_ptr = est.accessor<float, 1>();
	auto mask_ptr = mask.accessor<bool, 1>();
	auto old_guess = init;
	// auto new_guess = init;
	for(int i = acc_size-1; i>=0;i--)
	{
		mask_ptr[i] = old_guess < acc_ptr[i];
		if(old_guess < acc_ptr[i])
			old_guess *= (1+lammy*q);
		else
			old_guess *= (1-lammy*(1-q));
		est_ptr[i] = old_guess;
	}
	// return {{"mask", mask}, {"est", est}};
	return  std::make_tuple(mask, est);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("qe", &qe, "Quantile Estimation for Pytorch C++ Estimation");
}
