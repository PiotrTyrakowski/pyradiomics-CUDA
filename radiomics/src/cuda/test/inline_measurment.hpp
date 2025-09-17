#ifndef INLINE_MEASURMENT_HPP
#define INLINE_MEASURMENT_HPP

#ifdef ENABLE_TIME_MEASUREMENT

#include "test/framework.h"

#define START_MEASUREMENT(idx, ...) \
    time_measurement measurement_##idx; \
    PREPARE_DATA_MEASUREMENT(measurement_##idx, __VA_ARGS__)

#define END_MEASUREMENT(idx) \
    EndMeasurement(&measurement_##idx); \
    AddDataMeasurement(GetOngoingTest(), measurement_##idx)

#else
#define START_MEASUREMENT(idx, name) (void)0
#define END_MEASUREMENT(idx) (void)0
#endif // ENABLE_TIME_MEASUREMENT

#endif //INLINE_MEASURMENT_HPP
