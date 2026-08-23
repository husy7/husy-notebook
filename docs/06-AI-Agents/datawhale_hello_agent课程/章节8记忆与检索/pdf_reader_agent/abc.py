# Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""根据 PEP 3119 定义的抽象基类 (ABC)。"""


def abstractmethod(funcobj):
    """指示抽象方法的装饰器。

    要求元类是 ABCMeta 或其派生类。具有由 ABCMeta 派生的元类的类，
    除非其所有抽象方法都被重写，否则不能实例化。抽象方法可以使用任何
    标准的'super'调用机制进行调用。abstractmethod() 可用于声明属性和描述符的抽象方法。

    用法示例：

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    """
    funcobj.__isabstractmethod__ = True
    return funcobj


class abstractclassmethod(classmethod):
    """指示抽象类方法的装饰器。

    已弃用，请使用带有'abstractmethod'的'classmethod'：

        class C(ABC):
            @classmethod
            @abstractmethod
            def my_abstract_classmethod(cls, ...):
                ...

    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractstaticmethod(staticmethod):
    """指示抽象静态方法的装饰器。

    已弃用，请使用带有'abstractmethod'的'staticmethod'：

        class C(ABC):
            @staticmethod
            @abstractmethod
            def my_abstract_staticmethod(...):
                ...

    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractproperty(property):
    """指示抽象属性的装饰器。

    已弃用，请使用带有'abstractmethod'的'property'：

        class C(ABC):
            @property
            @abstractmethod
            def my_abstract_property(self):
                ...

    """

    __isabstractmethod__ = True


try:
    from _abc import (get_cache_token, _abc_init, _abc_register,
                      _abc_instancecheck, _abc_subclasscheck, _get_dump,
                      _reset_registry, _reset_caches)
except ImportError:
    from _py_abc import ABCMeta, get_cache_token
    ABCMeta.__module__ = 'abc'
else:
    class ABCMeta(type):
        """用于定义抽象基类 (ABC) 的元类。

        使用此元类创建 ABC。ABC 可以直接被继承，然后充当混合类。你还可以
        注册不相关的具体类（甚至是内置类）和不相关的 ABC 作为“虚拟子类”——
        这些类及其子类将被 issubclass() 函数视为注册 ABC 的子类，但注册的
        ABC 不会出现在它们的 MRO（方法解析顺序）中，由注册 ABC 定义的
        方法实现也不可调用的（即使通过 super()）。
        """
        def __new__(mcls, name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, name, bases, namespace, **kwargs)
            _abc_init(cls)
            return cls

        def register(cls, subclass):
            """注册 ABC 的虚拟子类。

            返回子类，以便作为类装饰器使用。
            """
            return _abc_register(cls, subclass)

        def __instancecheck__(cls, instance):
            """isinstance(instance, cls) 的重写实现。"""
            return _abc_instancecheck(cls, instance)

        def __subclasscheck__(cls, subclass):
            """issubclass(subclass, cls) 的重写实现。"""
            return _abc_subclasscheck(cls, subclass)

        def _dump_registry(cls, file=None):
            """调试辅助函数，用于打印 ABC 注册表。"""
            print(f"Class: {cls.__module__}.{cls.__qualname__}", file=file)
            print(f"Inv. counter: {get_cache_token()}", file=file)
            (_abc_registry, _abc_cache, _abc_negative_cache,
             _abc_negative_cache_version) = _get_dump(cls)
            print(f"_abc_registry: {_abc_registry!r}", file=file)
            print(f"_abc_cache: {_abc_cache!r}", file=file)
            print(f"_abc_negative_cache: {_abc_negative_cache!r}", file=file)
            print(f"_abc_negative_cache_version: {_abc_negative_cache_version!r}",
                  file=file)

        def _abc_registry_clear(cls):
            """清除注册表（用于调试或测试）。"""
            _reset_registry(cls)

        def _abc_caches_clear(cls):
            """清除缓存（用于调试或测试）。"""
            _reset_caches(cls)


def update_abstractmethods(cls):
    """重新计算抽象基类的抽象方法集合。

    如果一个类在其创建后实现了其某个抽象方法，那么该方法在调用此函数之前
    仍不会被认为已实现。或者，如果向类添加了新的抽象方法，只有在调用此函数
    之后，它才会被视为该类的抽象方法。

    应在对类进行任何使用之前调用此函数，通常是在向主题类添加方法的类装饰器中。

    返回 cls，以便作为类装饰器使用。

    如果 cls 不是 ABCMeta 的实例，则什么都不做。
    """
    if not hasattr(cls, '__abstractmethods__'):
        # 我们在这里检查 __abstractmethods__，因为 cls 可能是 C 实现或 Python 实现（特别是在测试期间），
        # 我们需要同时处理这两种情况。
        return cls

    abstracts = set()
    # 检查父类现有的抽象方法，只保留那些尚未实现的。
    for scls in cls.__bases__:
        for name in getattr(scls, '__abstractmethods__', ()):
            value = getattr(cls, name, None)
            if getattr(value, "__isabstractmethod__", False):
                abstracts.add(name)
    # 同时添加任何新添加的抽象方法。
    for name, value in cls.__dict__.items():
        if getattr(value, "__isabstractmethod__", False):
            abstracts.add(name)
    cls.__abstractmethods__ = frozenset(abstracts)
    return cls


class ABC(metaclass=ABCMeta):
    """辅助类，提供使用继承创建 ABC 的标准方式。
    """
    __slots__ = ()