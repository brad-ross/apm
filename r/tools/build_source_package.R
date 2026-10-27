#!/usr/bin/env Rscript
# Build a standalone source distribution without modifying scientific sources.
# Usage: Rscript r/tools/build_source_package.R --output-dir DIR [--manifest FILE]
main <- function() {
  for (pkg in c("digest", "jsonlite")) {
    if (!requireNamespace(pkg, quietly = TRUE)) stop("Install build dependency: ", pkg)
  }
  args <- commandArgs(trailingOnly = TRUE)
  options <- list("output-dir" = getwd(), manifest = NULL)
  while (length(args)) {
    key <- sub("^--", "", args[1L])
    if (!startsWith(args[1L], "--") || !key %in% names(options) || length(args) < 2L)
      stop("Usage: build_source_package.R --output-dir DIR [--manifest FILE]")
    options[[key]] <- args[2L]
    args <- args[-c(1L, 2L)]
  }
  script_arg <- grep("^--file=", commandArgs(), value = TRUE)
  if (length(script_arg) != 1L) stop("Run with Rscript")
  script <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
  root <- dirname(dirname(dirname(script)))
  output <- options[["output-dir"]]
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  output <- normalizePath(output, mustWork = TRUE)
  git <- function(...) {
    result <- system2("git", c("-C", shQuote(root), ...), stdout = TRUE)
    if (!is.null(attr(result, "status"))) stop("git command failed")
    result
  }
  revision <- git("rev-parse", "HEAD")
  if (length(revision) != 1L || !grepl("^[0-9a-f]{40}$", revision)) stop("Invalid Git SHA")
  hash <- function(path) digest::digest(file = path, algo = "sha256")
  write_json <- function(value, path) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    jsonlite::write_json(value, path, pretty = TRUE, auto_unbox = TRUE, null = "null")
  }
  work <- tempfile("apm-source-")
  dir.create(work)
  on.exit(unlink(work, recursive = TRUE), add = TRUE)
  stage <- file.path(work, "apm")
  dir.create(stage)
  excluded <- function(path) grepl(
    "(^|/)(tests?|\\.git|\\.Rproj\\.user)(/|$)|\\.(o|obj|so|dll|dylib|a|lib|rds|RData|rda)$|(^|/)symbols\\.rds$",
    path, ignore.case = TRUE)
  copy_sources <- function(from, to, core = FALSE) {
    files <- list.files(from, recursive = TRUE, all.files = TRUE, no.. = TRUE)
    files <- files[!excluded(files) & files != "tools/build_source_package.R"]
    if (core) files <- files[grepl("\\.(cpp|cc|c|h|hpp|hh|inc)$", files)]
    for (rel in files) {
      src <- file.path(from, rel)
      if (nzchar(Sys.readlink(src))) stop("Symlinks are not allowed: ", src)
      if (isTRUE(file.info(src)$isdir)) next
      dest <- file.path(to, rel)
      dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
      if (!file.copy(src, dest, overwrite = TRUE)) stop("Copy failed: ", src)
    }
  }
  copy_sources(file.path(root, "r"), stage)
  copy_sources(file.path(root, "core", "src"), file.path(stage, "src", "core"), core = TRUE)
  for (pair in list(c("r/LICENSE", "LICENSE"), c("LICENSE", "LICENSE.md"))) {
    if (!file.copy(file.path(root, pair[1L]), file.path(stage, pair[2L]), overwrite = TRUE))
      stop("Required license missing: ", pair[1L])
  }
  source_dir <- file.path(stage, "src")
  texts <- list.files(source_dir, recursive = TRUE, full.names = TRUE)
  texts <- texts[grepl("\\.(cpp|cc|c|h|hpp|hh|inc)$|Makevars(\\.win)?$", texts)]
  for (path in texts) {
    lines <- readLines(path, warn = FALSE)
    lines <- gsub("../../core/src", "core", lines, fixed = TRUE)
    writeLines(lines, path, useBytes = TRUE)
  }
  # R's default rule only discovers top-level sources. Explicitly include every
  # staged translation unit, including match attribution, on both platforms.
  cpp <- sort(list.files(source_dir, pattern = "\\.(cpp|cc|c)$", recursive = TRUE))
  objects <- sub("\\.(cpp|cc|c)$", ".o", cpp)
  for (name in c("Makevars", "Makevars.win")) {
    path <- file.path(source_dir, name)
    lines <- readLines(path, warn = FALSE)
    start <- grep("^OBJECTS[[:space:]]*=", lines)
    if (length(start) != 1L) stop("Expected one OBJECTS declaration in ", name)
    end <- start
    while (end < length(lines) && grepl("\\\\[[:space:]]*$", lines[end])) end <- end + 1L
    before <- if (start > 1L) lines[seq_len(start - 1L)] else character()
    after <- if (end < length(lines)) lines[seq.int(end + 1L, length(lines))] else character()
    writeLines(c(before, paste("OBJECTS =", paste(objects, collapse = " ")), after), path)
  }
  if (!all(c("core/target_params/match_attribution.o", "match_attribution_bindings.o") %in% objects))
    stop("Missing match attribution source")
  for (path in texts) {
    if (any(grepl("../../core/src", readLines(path, warn = FALSE), fixed = TRUE)))
      stop("External core path remains: ", path)
  }
  description <- read.dcf(file.path(stage, "DESCRIPTION"))
  package <- unname(description[1L, "Package"])
  version <- unname(description[1L, "Version"])
  staged_files <- sort(list.files(stage, recursive = TRUE, all.files = TRUE, no.. = TRUE))
  source_hashes <- setNames(vapply(file.path(stage, staged_files), hash, character(1L)), staged_files)
  patch <- git("diff", "--binary", "HEAD", "--", "r", "core/src", shQuote(":(exclude)r/tools/build_source_package.R"))
  provenance <- list(
    schema_version = 1L, package = package, version = version,
    repository = "https://github.com/brad-ross/apm", upstream_commit = revision,
    release_tag = NULL, release_url = NULL,
    historical_numerical_equivalence = "Not verified; no empirical analysis was rerun.",
    packaging = list(tool = "r/tools/build_source_package.R", tool_sha256 = hash(script),
      root_license_sha256 = hash(file.path(root, "LICENSE")),
      r_license_sha256 = hash(file.path(root, "r", "LICENSE")),
      tracked_scientific_patch_sha256 = digest::digest(paste(patch, collapse = "\n"), algo = "sha256", serialize = FALSE),
      transformations = c("Stage r/ and core/src under src/core", "Rewrite core includes and Makevars paths", "Derive both platform OBJECTS from staged translation units", "Exclude tests and compiled artifacts", "Include MIT notices")),
    staged_source_sha256 = as.list(source_hashes))
  write_json(provenance, file.path(stage, "inst", "APM-SOURCE.json"))
  old <- setwd(work)
  on.exit(setwd(old), add = TRUE)
  log <- system2(file.path(R.home("bin"), "R"),
    c("CMD", "build", "--no-build-vignettes", "--no-manual", shQuote(stage)),
    stdout = TRUE, stderr = TRUE)
  cat(paste(log, collapse = "\n"), "\n")
  if (!is.null(attr(log, "status"))) stop("R CMD build failed")
  archive <- paste0(package, "_", version, ".tar.gz")
  members <- utils::untar(archive, list = TRUE)
  if (any(excluded(members)) || "apm/tools/build_source_package.R" %in% members) stop("Source archive contains excluded files")
  if (!paste0(package, "/inst/APM-SOURCE.json") %in% members) stop("Missing installed provenance")
  destination <- file.path(output, archive)
  if (!file.copy(archive, destination, overwrite = TRUE)) stop("Unable to save archive")
  manifest <- list(schema_version = 1L, package = package, version = version,
    repository = provenance$repository, upstream_commit = revision,
    release_tag = NULL, release_url = NULL,
    source_tarball = paste0("vendor/", archive), source_sha256 = hash(destination),
    installed_provenance = "APM-SOURCE.json", packaging = provenance$packaging,
    build = list(r_version = as.character(getRversion()), platform = R.version$platform,
      command = "R CMD build --no-build-vignettes --no-manual <staged apm>",
      built_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
    verification = list(isolated_install = "pending", package_load = "pending",
      empirical_analysis = "not run", historical_numerical_equivalence = "not verified"))
  manifest_path <- options$manifest
  if (is.null(manifest_path)) manifest_path <- file.path(output, "apm-source.json")
  # Relative manifest paths are relative to the original invocation directory.
  if (!grepl("^(/|[A-Za-z]:[/\\\\])", manifest_path)) manifest_path <- file.path(old, manifest_path)
  write_json(manifest, manifest_path)
  message("Saved ", destination, "\nSHA-256: ", manifest$source_sha256,
    "\nManifest: ", manifest_path)
}
main()
