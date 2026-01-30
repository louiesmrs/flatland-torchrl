import pkg_resources

try:
    dist = pkg_resources.get_distribution('carbs')
    print('carbs version:', dist.version)
except pkg_resources.DistributionNotFound:
    print('carbs not installed')
