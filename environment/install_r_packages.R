args <- commandArgs(trailingOnly = TRUE)
lock_path <- if (length(args) >= 1L) args[[1L]] else "environment/r-requirements.lock"
expected_r <- "4.6.0"
cran_repo <- "https://cloud.r-project.org"

if (as.character(getRversion()) != expected_r) {
  stop(sprintf(
    "R version mismatch: expected %s, found %s",
    expected_r,
    as.character(getRversion())
  ))
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

needs_install <- vapply(seq_len(nrow(lock)), function(i) {
  installed_version <- tryCatch(
    as.character(utils::packageVersion(lock$package[[i]])),
    error = function(e) NA_character_
  )
  !identical(installed_version, lock$version[[i]])
}, logical(1L))

if (any(needs_install) && !requireNamespace("remotes", quietly = TRUE)) {
  install.packages(
    "remotes",
    repos = cran_repo,
    dependencies = c("Depends", "Imports", "LinkingTo")
  )
}

for (i in seq_len(nrow(lock))) {
  package_name <- lock$package[[i]]
  expected_version <- lock$version[[i]]
  installed_version <- tryCatch(
    as.character(utils::packageVersion(package_name)),
    error = function(e) NA_character_
  )

  if (needs_install[[i]]) {
    remotes::install_version(
      package = package_name,
      version = expected_version,
      repos = cran_repo,
      upgrade = "never",
      dependencies = c("Depends", "Imports", "LinkingTo")
    )
  }
}

for (i in seq_len(nrow(lock))) {
  package_name <- lock$package[[i]]
  expected_version <- lock$version[[i]]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop(sprintf("R package failed to load: %s", package_name))
  }
  installed_version <- as.character(utils::packageVersion(package_name))
  if (!identical(installed_version, expected_version)) {
    stop(sprintf(
      "R package version mismatch after installation: %s expected %s, found %s",
      package_name,
      expected_version,
      installed_version
    ))
  }
  message(sprintf("%s=%s", package_name, installed_version))
}

message("R_PACKAGE_LOCK_CHECK=PASS")
