#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

#include <chrono>
#include <iostream>
#include <cstdlib>

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
result_t since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

#ifdef IS_TESTING
bool _show_timer = true;
#else
bool _show_timer = std::getenv("SHOW_TIMER") != nullptr;
#endif

#define START_TIMER() std::chrono::steady_clock::time_point _timer_start = std::chrono::steady_clock::now();
#define RECORD_TIME(msg) { if (_show_timer) std::cerr << msg << ": " << since(_timer_start).count() << "ms" << std::endl; _timer_start = std::chrono::steady_clock::now(); }


#endif // UTILS_TIMER_H