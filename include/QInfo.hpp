#pragma once

enum class QInfo {
    Zero,     // Q = 0
    Diagonal, // Q = diag(q)
    General   // general SPD, use LLT
};