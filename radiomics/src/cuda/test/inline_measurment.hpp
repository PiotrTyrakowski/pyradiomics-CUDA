#ifndef INLINE_MEASURMENT_HPP
#define INLINE_MEASURMENT_HPP

#ifdef ENABLE_TIME_MEASUREMENT

#include "test/framework.h"

#define START_MEASUREMENT(idx, name) \
    StartMeasurement(idx, name)

#define END_MEASUREMENT(idx) \
    EndMeasurement(idx)

#else
#define START_MEASUREMENT(idx, name) (void)0
#define END_MEASUREMENT(idx) (void)0
#endif // ENABLE_TIME_MEASUREMENT

#endif //INLINE_MEASURMENT_HPP
