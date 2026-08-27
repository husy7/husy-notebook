
#1.
#最基本的形式是直接在函数参数和返回值上使用标准类型，如 float、str 等。
def surface_area_of_cube(edge_length: float) -> str:
    return f"The surface area of the cube is {6 * edge_length ** 2}."

#2.
#类型别名 (Type Alias)
#类型别名可以为复杂的类型签名起一个更简短、易于理解的名字。

#type Vector = list[float] #使用 type 语句 (Python 3.12+):
Vector = list[float] #使用简单赋值 (兼容旧版本):
def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

# 类型检查通过
new_vector = scale(2.0, [1.0, -4.2, 5.4])


#更复杂的示例: (Python 3.12+)
'''
from collections.abc import Sequence

type ConnectionOptions = dict[str, str]
type Address = tuple[str, int]
type Server = tuple[Address, ConnectionOptions]

def broadcast_message(message: str, servers: Sequence[Server]) -> None:
    ...
'''


#3. 创建新类型 (NewType)
#NewType 用于创建原类型的“子类型”，帮助静态类型检查器捕捉逻辑错误。

#定义与使用:

from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)

def get_user_name(user_id: UserId) -> str:
    ...

# 类型检查通过
user_a = get_user_name(UserId(42351))
# 类型检查失败：-1 不是 UserId 类型
user_b = get_user_name(-1)

#基于 NewType 创建新类型:


ProUserId = NewType('ProUserId', UserId)

#4. 标注可调用对象 (Callable)
#使用 collections.abc.Callable 来标注函数或其它可调用对象。

#基本用法:

from collections.abc import Callable

def feeder(get_next_item: Callable[[], str]) -> None:
    ...

def async_query(on_success: Callable[[int], None],
                on_error: Callable[[int, Exception], None]) -> None:
    ...

#接受任意参数:

def concat(x: str, y: str) -> str:
    return x + y

# 可接受任何参数列表
x: Callable[..., str] = str
x = concat  # 同样可以

#5. 泛型 (Generics)
#泛型允许你定义适用于多种类型的函数或类。

#内置容器的泛型:


from collections.abc import Mapping, Sequence

class Employee:
    ...

# 表明序列中的元素都必须是 Employee 的实例
def notify_by_email(employees: Sequence[Employee], overrides: Mapping[str, str]) -> None:
    ...

#泛型函数 (使用 def first[T] 语法, Python 3.12+):

'''
from collections.abc import Sequence

def first[T](l: Sequence[T]) -> T:
    return l[0]
'''
#泛型函数 (使用 TypeVar 工厂):


from collections.abc import Sequence
from typing import TypeVar

U = TypeVar('U')

def second(l: Sequence[U]) -> U:
    return l[1]

#🏗️ 用户定义的泛型类型
#你可以创建自己的泛型类。

#使用 class LoggedVar[T] 语法 (Python 3.12+):

'''
from logging import Logger

class LoggedVar[T]:
    def __init__(self, value: T, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.value = value

    def set(self, new: T) -> None:
        self.log('Set ' + repr(self.value))
        self.value = new

    def get(self) -> T:
        self.log('Get ' + repr(self.value))
        return self.value

    def log(self, message: str) -> None:
        self.logger.info('%s: %s', self.name, message)

 '''

#兼容旧版本的写法 (显式继承 Generic):


from typing import TypeVar, Generic

T = TypeVar('T')

class LoggedVar(Generic[T]):
    ...

#6 Any 类型
#Any 是一种特殊的类型，它表示动态类型，可以兼容所有类型。


from typing import Any

a: Any = None
a = []     # 可以
a = 2      # 可以

s: str = ''
s = a      # 可以

def foo(item: Any) -> int:
    # 类型检查通过，因为 item 可以被认为是任何类型
    item.bar()
    ...