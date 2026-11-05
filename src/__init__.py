"""AML Transaction Monitoring ML Package."""

# Compatibility shim for models serialized with older scikit-learn versions
import sklearn.compose._column_transformer as _ct_module

if not hasattr(_ct_module, "_RemainderColsList"):

    class _RemainderColsList(list):
        """Compatibility shim for scikit-learn models serialized on older versions."""

        pass

    _ct_module._RemainderColsList = _RemainderColsList
