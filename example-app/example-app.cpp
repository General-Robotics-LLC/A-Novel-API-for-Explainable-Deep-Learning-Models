#include <torch/torch.h>
#include <iostream>

using std::cout, std::cin, std::endl;

class CustomLogSoftmaxImpl : public torch::nn::LogSoftmaxImpl {

    private:
        torch::Tensor activation_values;
        torch::Tensor activation_derivatives;
    public:
        using torch::nn::LogSoftmaxImpl::LogSoftmaxImpl;

        torch::Tensor forward(const torch::Tensor &input) {
            auto activation_values = torch::nn::functional::log_softmax(input, 1);
            cout << this->activation_values.sizes() << activation_values.sizes() << endl;
            this->activation_values = torch::cat({this->activation_values, activation_values}, 0);
            return activation_values;
        }

        void ResetAttributes() {
            this->activation_values = torch::empty({0, 16});
            this->activation_derivatives = torch::empty({0, 16});
        }

        torch::Tensor GetActivationValues() {
            return this->activation_values;
        }
        void SetActivationValues(const torch::Tensor activation_values) {
            this->activation_values = activation_values;
        }

        torch::Tensor GetActivationDerivatives() {
            return this->activation_derivatives;
        }
        void SetActivationDerivatives(const torch::Tensor activation_derivatives) {
            this->activation_derivatives = activation_derivatives;
        }
};

class CustomLinearImpl : public torch::nn::LinearImpl {
    private:
        double _init_new_attribute = 0.0;
        double new_attribute = _init_new_attribute;
    public:
        using torch::nn::LinearImpl::LinearImpl;

        void ResetAttributes() {
            this->new_attribute = _init_new_attribute;
        }
        void printNewAttribute() const {
            std::cout << "New Attribute: " << new_attribute << std::endl;
        }
};

TORCH_MODULE(CustomLinear);
TORCH_MODULE(CustomLogSoftmax);

struct Net : torch::nn::Module {

    public:
        CustomLinear l1;
        torch::nn::Linear l2;
        CustomLogSoftmax act2;

    public:
        Net() :
            l1(torch::nn::LinearOptions(4, 8).bias(true)),
            l2(torch::nn::LinearOptions(8, 16).bias(true)),
            act2(torch::nn::LogSoftmaxOptions(1))
        {
            register_module("l1", l1);
            register_module("l2", l2);
            register_module("act2", act2);
        }


        torch::Tensor forward(torch::Tensor x) {
            x = l1->forward(x);
            x = torch::sigmoid(x);

            x = l2->forward(x);
            x = act2->forward(x);
            return x;
        }


        void ResetAttributes() {
            l1->ResetAttributes();
            act2->ResetAttributes();
        }



        // print gradient
        void print_gradients() {
            for (auto& p : named_parameters()) {
                std::cout << p.key() << ":\n" << p.value().grad() << std::endl;
            }
        }
};




int main() {
    int num_epochs = 100;
    int batch_size = 20;
    int num_classes = 16;
    int num_data = 16;

    auto input_tensor = torch::tensor({{0, 0, 0, 0},
                        {0, 0, 0, 1},
                        {0, 0, 1, 0},
                        {0, 0, 1, 1},
                        {0, 1, 0, 0},
                        {0, 1, 0, 1},
                        {0, 1, 1, 0},
                        {0, 1, 1, 1},
                        {1, 0, 0, 0},
                        {1, 0, 0, 1},
                        {1, 0, 1, 0},
                        {1, 0, 1, 1},
                        {1, 1, 0, 0},
                        {1, 1, 0, 1},
                        {1, 1, 1, 0},
                        {1, 1, 1, 1}}, torch::dtype(torch::kFloat));
    input_tensor.set_requires_grad(true);

    auto target_tensor = torch::arange(0, 16);



    Net net;
    torch::optim::SGD optimizer(net.parameters(), /*lr=*/10);

    cout << "Training Start" << endl;
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
      optimizer.zero_grad();
      net.ResetAttributes();

      torch::Tensor prediction = net.forward(input_tensor);
      
      torch::Tensor loss = torch::nll_loss(prediction, target_tensor);

      loss.backward();

      optimizer.step();

        std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;
    }


    cout << net.act2->GetActivationValues().sizes() << endl;

    cout << "Training Finished" << endl;



    net.ResetAttributes();

    auto pred_input = torch::tensor({{0, 0, 0, 0},
                        {0, 0, 0, 1},
                        {0, 0, 1, 0},
                        {0, 0, 1, 1},
                        {0, 1, 0, 0},
                        {0, 1, 0, 1},
                        {0, 1, 1, 0},
                        {0, 1, 1, 1},
                        {1, 0, 0, 0},
                        {1, 0, 0, 1},
                        {1, 0, 1, 0},
                        {1, 0, 1, 1},
                        {1, 1, 0, 0},
                        {1, 1, 0, 1},
                        {1, 1, 1, 0},
                        {1, 1, 1, 1}}, torch::dtype(torch::kFloat));

    auto prediction = net.forward(pred_input);

    auto pred = prediction.argmax(1);
    cout<<pred<<endl;

    cout << net.act2->GetActivationValues().sizes() << endl;

    system("pause");

    return 0;
}