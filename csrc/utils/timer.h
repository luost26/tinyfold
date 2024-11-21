#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

#include <chrono>
#include <iostream>

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
result_t since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

#define START_TIMER() std::chrono::steady_clock::time_point _timer_start = std::chrono::steady_clock::now();
#define RECORD_TIME(msg) { std::cerr << msg << ": " << since(_timer_start).count() << "ms" << std::endl; _timer_start = std::chrono::steady_clock::now(); }


#endif // UTILS_TIMER_H