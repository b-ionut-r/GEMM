#include "step_1.cu"
#include "step_2.cu"
#include "step_3.cu"


int main() {
    int n;
    std::cout << "Profile for version: ";
    std::cin >> n;
    switch (n) {
        case 1: step_1();
        case 2: step_2();
        case 3: step_3();
        default: ;
    }
}