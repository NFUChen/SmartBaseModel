from typing import Optional, TypeVar, Generic, Callable

T = TypeVar("T")


class Observable(Generic[T]):
    """
    Implements an Observable class that allows subscribing and unsubscribing to events.
    The Observable class maintains a list of subscriber callbacks and provides methods to subscribe, unsubscribe, and emit events to those subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[T], None]] = []

    def subscribe(self, subscriber: Callable[[T], None]) -> None:
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Callable[[T], None]) -> None:
        self._subscribers.remove(subscriber)

    def emit(self, value: T) -> None:
        for subscriber in self._subscribers:
            subscriber(value)


class BehaviorSubject(Generic[T]):
    """
    A BehaviorSubject is a type of Observable that emits its current value to new subscribers when they subscribe. 
    It maintains the last emitted value and immediately emits it to any new subscriber.
    
    The BehaviorSubject class provides the following methods:
    
    - `__init__()`: Initializes a new BehaviorSubject instance with an optional initial value.
    - `as_observable()`: Returns an Observable view of the BehaviorSubject.
    - `subscribe(subscriber)`: Subscribes the given callable to the BehaviorSubject. The callable will be called with the next emitted value.
    - `next(value)`: Emits the given value to all subscribed callables.
    - `value`: Returns the last emitted value, or None if no value has been emitted yet.
    """
        
    def __init__(self) -> None:
        self._value: Optional[T] = None
        self._observable = Observable[T]()

    def as_observable(self) -> Observable[T]:
        return self._observable

    def subscribe(self, subscriber: Callable[[T], None]) -> None:
        self._observable._subscribers.append(subscriber)

    def next(self, value: T) -> None:
        self._value = value
        self._observable.emit(self._value)

    @property
    def value(self) -> Optional[T]:
        return self._value


if __name__ == "__main__":
    # Example usage:

    def subscriber(value: int) -> None:
        print("Received:", value)

    subject = BehaviorSubject[int]()
    observable = subject.as_observable()
    observable.subscribe(subscriber)

    subject.next(1)  # Output: Received: 1
    subject.next(2)  # Output: Received: 2
