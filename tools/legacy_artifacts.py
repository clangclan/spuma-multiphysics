"""Read immutable evidence recorded before the generic ReactiveFoam names.

These aliases apply to archived metadata, not to runtime shared-library loading.
"""
BINARY_PATH_ALIASES = {
    'lib/libpintleReactiveBackend.so': 'lib/libreactiveBackend.so',
    'lib/libpintleReactiveTransport.so': 'lib/libreactiveTransport.so',
}


def canonical_binary_hashes(values):
    result = {}
    for path, digest in values.items():
        path = BINARY_PATH_ALIASES.get(path, path)
        if path in result and result[path] != digest:
            raise ValueError('Conflicting historical/current binary hash: ' + path)
        result[path] = digest
    return result


def strip_transport_namespace(name):
    return name.replace('ReactiveTransport::', '').replace('PintleTransport::', '')
