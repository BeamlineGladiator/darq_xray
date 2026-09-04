"""PySide6 desktop application.

This subpackage is the *only* place Qt is imported. It calls into the Qt-free
``darq_xray`` core; the core never imports anything from here.
"""
