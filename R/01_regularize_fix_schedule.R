# Public-safe telemetry regularization utilities.
#
# This script keeps the reusable method while avoiding study-specific paths,
# animal IDs, and unpublished split manifests.

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readr)
})

nearest_target_time <- function(timestamp, target_hours = c(0, 8, 16)) {
  hour_decimal <- hour(timestamp) + minute(timestamp) / 60 + second(timestamp) / 3600
  target_hours[which.min(abs(hour_decimal - target_hours))]
}

regularize_gps <- function(
    data,
    id_col = "animal_id",
    time_col = "timestamp",
    target_hours = c(0, 8, 16),
    min_fixes_per_day = 2) {
  stopifnot(id_col %in% names(data), time_col %in% names(data))

  data %>%
    mutate(
      .timestamp = ymd_hms(.data[[time_col]], tz = "UTC"),
      .date = as.Date(.timestamp),
      .target_hour = vapply(.timestamp, nearest_target_time, numeric(1), target_hours = target_hours),
      .target_time = as.POSIXct(.date, tz = "UTC") + hours(.target_hour),
      .offset_seconds = abs(as.numeric(difftime(.timestamp, .target_time, units = "secs")))
    ) %>%
    group_by(.data[[id_col]], .date, .target_hour) %>%
    slice_min(.offset_seconds, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    group_by(.data[[id_col]], .date) %>%
    filter(n() >= min_fixes_per_day) %>%
    ungroup() %>%
    arrange(.data[[id_col]], .target_time) %>%
    mutate(regularized_timestamp = .target_time) %>%
    select(-.timestamp, -.date, -.target_hour, -.target_time, -.offset_seconds)
}

add_segments <- function(
    data,
    id_col = "animal_id",
    time_col = "regularized_timestamp",
    max_gap_hours = 16) {
  stopifnot(id_col %in% names(data), time_col %in% names(data))

  data %>%
    mutate(.timestamp = ymd_hms(.data[[time_col]], tz = "UTC")) %>%
    arrange(.data[[id_col]], .timestamp) %>%
    group_by(.data[[id_col]]) %>%
    mutate(
      gap_hours = as.numeric(difftime(.timestamp, lag(.timestamp), units = "hours")),
      segment_id = cumsum(is.na(gap_hours) | gap_hours > max_gap_hours)
    ) %>%
    ungroup() %>%
    select(-.timestamp)
}

write_regularized_outputs <- function(
    input_csv,
    output_csv,
    target_hours = c(0, 8, 16),
    min_fixes_per_day = 2,
    max_gap_hours = 16) {
  read_csv(input_csv, show_col_types = FALSE) %>%
    regularize_gps(target_hours = target_hours, min_fixes_per_day = min_fixes_per_day) %>%
    add_segments(max_gap_hours = max_gap_hours) %>%
    write_csv(output_csv)
}
