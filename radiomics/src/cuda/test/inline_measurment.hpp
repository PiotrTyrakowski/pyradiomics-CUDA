#ifndef INLINE_MEASURMENT_HPP
#define INLINE_MEASURMENT_HPP

#ifdef ENABLE_TIME_MEASUREMENT

#include "test/framework.h"

#define START_MEASUREMENT(name) \
    StartMeasurement(name)

#define END_MEASUREMENT(name) \
    EndMeasurement(name)

#else
#define START_MEASUREMENT(name) (void)0
#define END_MEASUREMENT(name) (void)0
#endif // ENABLE_TIME_MEASUREMENT

#endif //INLINE_MEASURMENT_HPP
