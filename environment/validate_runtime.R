args <- commandArgs(trailingOnly = TRUE)
lock_path <- if (length(args) >= 1L) args[[1L]] else "environment/r-requirements.lock"
expected_r <- "4.6.0"

actual_r <- as.character(getRversion())
if (!identical(actual_r, expected_r)) {
  stop(sprintf("R version mismatch: expected %s, found %s", expected_r, actual_r))
}

lock <- read.delim(
  lock_path,
  header = TRUE,
  stringsAsFactors = FALSE,
  colClasses = "character",
  check.names = FALSE
)
if (!identical(names(lock), c("package", "version")) || nrow(lock) == 0L) {
  stop("Invalid R package lock file: expected non-empty package and version columns")
}

message(sprintf("R=%s", actual_r))
for (i in seq_len(nrow(lock))) {
  package_name <- lock$package[[i]]
  expected_version <- lock$version[[i]]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop(sprintf("Required R package is unavailable: %s", package_name))
  }
  installed_version <- as.character(utils::packageVersion(package_name))
  if (!identical(installed_version, expected_version)) {
    stop(sprintf(
      "R package version mismatch: %s expected %s, found %s",
      package_name,
      expected_version,
      installed_version
    ))
  }
  message(sprintf("%s=%s", package_name, installed_version))
}

message("R_RUNTIME_CHECK=PASS")
