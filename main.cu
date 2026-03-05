#include <iostream>
#include "steps.h"

int main() {
    int n;
    std::cout << "Profile for version: ";
    std::cin >> n;
    switch (n) {
        case 1: step_1(); break;
        case 2: step_2(); break;
        case 3: step_3(); break;
        case 4: step_4(); break;
        case 5: step_5(); break;
        default: ;
    }
}